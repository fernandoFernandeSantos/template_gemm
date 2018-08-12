################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/GEMM_kernels.cu \
../src/matrixMul.cu 

CPP_SRCS += \
../src/GEMM.cpp 

OBJS += \
./src/GEMM.o \
./src/GEMM_kernels.o \
./src/matrixMul.o 

CU_DEPS += \
./src/GEMM_kernels.d \
./src/matrixMul.d 

CPP_DEPS += \
./src/GEMM.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -I"/usr/local/cuda-9.2/samples/0_Simple" -I"/usr/local/cuda-9.2/samples/common/inc" -I"/home/fernando/cuda-workspace/template_gemm" -G -g -O0 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -I"/usr/local/cuda-9.2/samples/0_Simple" -I"/usr/local/cuda-9.2/samples/common/inc" -I"/home/fernando/cuda-workspace/template_gemm" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -I"/usr/local/cuda-9.2/samples/0_Simple" -I"/usr/local/cuda-9.2/samples/common/inc" -I"/home/fernando/cuda-workspace/template_gemm" -G -g -O0 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -I"/usr/local/cuda-9.2/samples/0_Simple" -I"/usr/local/cuda-9.2/samples/common/inc" -I"/home/fernando/cuda-workspace/template_gemm" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


