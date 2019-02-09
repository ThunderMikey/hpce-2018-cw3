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
#define __CL_ENALBE_EXCEPTIONS
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
void kernel_xy(uint32_t x, uint32_t y, uint32_t w,
    float outer, float inner,
    const float *states, const uint32_t *props,
    float *buffer){
    unsigned index=y*w + x;
    
    if((props[index] & Cell_Fixed) || (props[index] & Cell_Insulator)){
      // Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
      buffer[index]=states[index];
    }else{
      float contrib=inner;
      float acc=inner*states[index];
      
      // Cell above
      if(! (props[index-w] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index-w];
      }
      
      // Cell below
      if(! (props[index+w] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index+w];
      }
      
      // Cell left
      if(! (props[index-1] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index-1];
      }
      
      // Cell right
      if(! (props[index+1] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index+1];
      }
      
      // Scale the accumulate value by the number of places contributing to it
      float res=acc/contrib;
      // Then clamp to the range [0,1]
      res=std::min(1.0f, std::max(0.0f, res));
      buffer[index] = res;
      
    } // end of if(insulator){ ... } else {

}

	
//! Reference world stepping program
/*! \param dt Amount to step the world by.  Note that large steps will be unstable.
	\param n Number of times to step the world
	\note Overall time increment will be n*dt
*/
void StepWorldV3OpenCL(world_t &world, float dt, unsigned n)
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

  std::string kernelSource = LoadSource("step_world_v3_kernel.cl");
  cl::Program::Sources sources;
  sources.push_back(std::make_pair(
        kernelSource.c_str(),
        kernelSource.size()+1
        )
      );

  std::cerr<<kernelSource.c_str()<<std::endl;
  std::cerr<<kernelSource.size()<<std::endl;
  std::cerr<<kernelSource.length()<<std::endl;
  cl::Program program(context, sources);
  try{
    program.build(devices);
    std::cerr<<"Program built\n";
  }catch(...){
    for(unsigned i=0; i<no_devices; i++){
      std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
      std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
    }
    throw;
  }

	unsigned w=world.w, h=world.h;
	
	float outer=world.alpha*dt;		// We spread alpha to other cells per time
	float inner=1-outer/4;				// Anything that doesn't spread stays
	
	// This is our temporary working space
	std::vector<float> buffer(w*h);

	
	for(unsigned t=0;t<n;t++){
		for(unsigned y=0;y<h;y++){
			for(unsigned x=0;x<w;x++){
        kernel_xy(x, y, w,
            outer, inner,
            &world.state[0], (const uint32_t *)&world.properties[0],
            &buffer[0]);
			}  // end of for(x...
		} // end of for(y...
		
		// All cells have now been calculated and placed in buffer, so we replace
		// the old state with the new state
		std::swap(world.state, buffer);
		// Swapping rather than assigning is cheaper: just a pointer swap
		// rather than a memcpy, so O(1) rather than O(w*h)
	
		world.t += dt; // We have moved the world forwards in time
		
	} // end of for(t...
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
		hpce::yc12015::StepWorldV3OpenCL(world, dt, n);
		
		hpce::SaveWorld(std::cout, world, binary);
	}catch(const std::exception &e){
		std::cerr<<"Exception : "<<e.what()<<std::endl;
		return 1;
	}
		
	return 0;
}
