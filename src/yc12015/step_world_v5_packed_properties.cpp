#include "heat.hpp"

#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <fstream>
#include <streambuf>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

namespace hpce{
  namespace yc12015{

std::string LoadSource(const char *fileName){
  const char *v = getenv("HPCE_CL_SRC_DIR");
  std::string baseDir = v? v: "src/yc12015";

  std::string fullName = baseDir+"/"+fileName;

  std::ifstream src(fullName, std::ios::in | std::ios::binary);
  if(!src.is_open()){
    throw std::runtime_error("LoadSource: Couldn't load cl file from '"+
        fullName+"'.\n");
  }
  return std::string(
      (std::istreambuf_iterator<char>(src)), // extra brackets?
      std::istreambuf_iterator<char>()
      );
}

//! Reference world stepping program
/*! \param dt Amount to step the world by.  Note that large steps will be unstable.
	\param n Number of times to step the world
	\note Overall time increment will be n*dt
*/
void StepWorldV4DoubleBuffered(world_t &world, float dt, unsigned n)
{

  // show platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::vector<cl::Platform>::size_type no_platforms = platforms.size();
  if(no_platforms == 0){
    throw std::runtime_error("No OpenCL plaforms found.\n");
  }
  else{
    std::cerr<<"Found "<<no_platforms<<" platforms"<<std::endl;
  }
  for(unsigned i=0; i<no_platforms; i++){
    std::string vendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
    std::cerr<<"\tPlatform "<<i<<" : "<<vendor<<std::endl;
  }
  // get from env
  const char *v = getenv("HPCE_SELECT_PLATFORM");
  // default platform is 0
  int selectedPlatform = v? atoi(v): 0;
  std::cerr<<"Choosing platform "<<selectedPlatform<<std::endl;
  cl::Platform platform = platforms.at(selectedPlatform);

  // show devices
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  std::vector<cl::Device>::size_type no_devices = devices.size();
  if(no_devices == 0){
    throw std::runtime_error("No OpenCL devices found.\n");
  }
  else{
    std::cerr<<"Found "<<no_devices<<" devies"<<std::endl;
  }
  // show devices
  for(unsigned i=0; i<no_devices; i++){
    std::string deviceName = devices[i].getInfo<CL_DEVICE_NAME>();
    std::cerr<<"\tDevice "<<i<<" : "<<deviceName<<std::endl;
  }
  // get from env
  const char *u = getenv("HPCE_SELECT_DEVICE");
  // default platform is 0
  int selectedDevice = u? atoi(u): 0;
  std::cerr<<"Choosing device "<<selectedDevice<<std::endl;
  cl::Device device = devices.at(selectedDevice);

  // create context
  cl::Context context(devices);
  //--------------------
  // build cl program
  std::string kernelSource = LoadSource("step_world_v5_packed_properties.cl");
  cl::Program::Sources sources;
  sources.push_back(std::make_pair(
        kernelSource.c_str(),
        kernelSource.size()+1
        )
      );

  cl::Program program(context, sources);
  try{
    program.build(devices);
  }catch(...){
    for(unsigned i=0; i<no_devices; i++){
      std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
      std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
    }
    throw;
  }

  // ----------------
  // allocate buffers
  size_t cbBuffer = 4*world.w*world.h;
  cl::Buffer buffProperties(context, CL_MEM_READ_ONLY, cbBuffer);
  cl::Buffer buffState(context, CL_MEM_READ_WRITE, cbBuffer);
  cl::Buffer buffBuffer(context, CL_MEM_READ_WRITE, cbBuffer);


	unsigned w=world.w, h=world.h;

	float outer=world.alpha*dt;		// We spread alpha to other cells per time
	float inner=1-outer/4;				// Anything that doesn't spread stays

	// This is our temporary working space
  // no longer need buffer in main memory
  // as it exists in GPU global shared memory
	//std::vector<float> buffer(w*h);

  // ---------------
  // setting kernel parameters
  cl::Kernel kernel(program, "kernel_xy");
  kernel.setArg(0, inner);
  kernel.setArg(1, outer);
  kernel.setArg(2, buffProperties);

  // ---------------
  // create command queue
  cl::CommandQueue queue(context, device);

  // copy mem buffers
  cl::Event evCopiedState;
  queue.enqueueWriteBuffer(
      buffState,
      CL_FALSE,
      0,
      cbBuffer,
      &world.state[0],
      NULL,
      &evCopiedState
      );
  // define kernel exe params
  cl::NDRange offset(0, 0);
  cl::NDRange globalSize(w, h);
  cl::NDRange localSize = cl::NullRange;

  // kernel no longer depends on buffState as it is changing every loop
  std::vector<cl::Event> kernelDependencies;
  cl::Event evExecutedKernel;

  // pack neighbour properties into uint
  // packed properties definition
  // this:  1-0
  // above: 3-2
  // below: 5-4
  // left:  7-6
  // right: 9-8
  std::vector<uint32_t> packedProps(w*h, 0);

  for(unsigned y=0; y<h; y++){
    for(unsigned x=0; x<w; x++){
      unsigned idx = y*w+x;
      uint32_t& thisProp = packedProps[idx];
      thisProp = world.properties[idx];
      if(!(thisProp & Cell_Fixed || thisProp & Cell_Insulator)){
        // above
        if(world.properties[idx-w] & Cell_Insulator){
          thisProp += (Cell_Insulator << 2);
        }
        // below
        if(world.properties[idx+w] & Cell_Insulator){
          thisProp += (Cell_Insulator << 4);
        }
        // left
        if(world.properties[idx-1] & Cell_Insulator){
          thisProp += (Cell_Insulator << 6);
        }
        // right
        if(world.properties[idx+1] & Cell_Insulator){
          thisProp += (Cell_Insulator << 8);
        }
      }
    }
  }

  // -------------------
  // copy over fixed data: packed properties
  queue.enqueueWriteBuffer(
      buffProperties,
      CL_TRUE, 0, cbBuffer,
      &packedProps[0]
      );

  // set barrier
  queue.enqueueBarrierWithWaitList();

	for(unsigned t=0;t<n;t++){
    // set args for every loop
    kernel.setArg(3, buffState);
    kernel.setArg(4, buffBuffer);
    queue.enqueueNDRangeKernel(
        kernel,
        offset,
        globalSize,
        localSize,
        &kernelDependencies,
        &evExecutedKernel
        );

    queue.enqueueBarrierWithWaitList();
		// All cells have now been calculated and placed in buffer, so we replace
		// the old state with the new state
		std::swap(buffState, buffBuffer);
		// Swapping rather than assigning is cheaper: just a pointer swap
		// rather than a memcpy, so O(1) rather than O(w*h)
	
		world.t += dt; // We have moved the world forwards in time
		
	} // end of for(t...
  // copy the results back
  std::vector<cl::Event> copyBackDependencies(1, evExecutedKernel);
  queue.enqueueReadBuffer(
      buffState,
      CL_TRUE,
      0,
      cbBuffer,
      &world.state[0],
      &copyBackDependencies
      );
}

}; // namespace yc12015
}; // namepspace hpce

int main(int argc, char *argv[])
{
	float dt=0.1;
	unsigned n=1;
	bool binary=false;
	
	if(argc>1){
		dt=(float)strtod(argv[1], NULL);
	}
	if(argc>2){
		n=atoi(argv[2]);
	}
	if(argc>3){
		if(atoi(argv[3]))
			binary=true;
	}
	
	try{
		hpce::world_t world=hpce::LoadWorld(std::cin);
		std::cerr<<"Loaded world with w="<<world.w<<", h="<<world.h<<std::endl;
		
		std::cerr<<"Stepping by dt="<<dt<<" for n="<<n<<std::endl;
		hpce::yc12015::StepWorldV4DoubleBuffered(world, dt, n);
		
		hpce::SaveWorld(std::cout, world, binary);
	}catch(const std::exception &e){
		std::cerr<<"Exception : "<<e.what()<<std::endl;
		return 1;
	}
		
	return 0;
}
