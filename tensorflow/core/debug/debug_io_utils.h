#ifndef TENSORFLOW_DEBUG_DEBUG_IO_UTILS_H_
#define TENSORFLOW_DEBUG_DEBUG_IO_UTILS_H_

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

// Tensor ReadTensorFromProtoFile(const string& filename);
Tensor ReadTensorFromProtoFile(const string& filename);

}  // namespace tensorflow

#endif  // TENSORFLOW_DEBUG_DEBUG_IO_UTILS_H_
