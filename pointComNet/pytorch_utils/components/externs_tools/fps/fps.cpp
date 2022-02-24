#include <Python.h>
#include <torch/script.h>
#include "fps_cuda.h"
#include "fps_cpu.h"

torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, torch::Tensor ratio,
                  bool random_start) {
  if (src.device().is_cuda()) {
    return fps_cuda(src, ptr, ratio, random_start);
  }
  else
  {
    return fps_cpu(src, ptr, ratio, random_start);
  }
}

static auto registry =
    torch::RegisterOperators().op("my_torch_cluster::fps", &fps);