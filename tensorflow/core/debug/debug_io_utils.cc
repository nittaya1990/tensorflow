#include "tensorflow/core/debug/debug_io_utils.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/record_reader.h"

// int main(int argc, char* argv[]) {
namespace tensorflow {

// Tensor ReadTensorFromProtoFile(const string& filename) {
Tensor ReadTensorFromProtoFile(const string& filename) {
  // if (argc != 2) {
  //   std::cout << "Usage: debug_util <TENSOR_PROTO_FILENAME>" << std::endl;
  //   return 1;
  // }
  // std::string filename = argv[1];

  Env* env = Env::Default();

  // std::cout << "filename = " << filename << std::endl;

  std::unique_ptr<RandomAccessFile> tp_file;
  env->NewRandomAccessFile(filename, &tp_file);
  io::RecordReader record_reader(tp_file.get());

  string readout;
  uint64 offset = 0;
  record_reader.ReadRecord(&offset, &readout);

  TensorProto tensor_proto_2;
  tensor_proto_2.ParseFromString(readout);
  Tensor tensor2; 
  if (!tensor2.FromProto(tensor_proto_2)) {
    std::cerr << "ERROR: FromProto() failed" << std::endl;  // DEBUG
  } else {
    TensorShape tensor_shape = tensor2.shape();
  
    std::cout << "Readout tensor: " << tensor2.SummarizeValue(1000) << std::endl;  // DEBUG
    std::cout << "  shape: " << tensor_shape.DebugString() << std::endl;  // DEBUG
    std::cout << "  shape: " << tensor_shape.DebugString() << std::endl;  // DEBUG
  }

  return tensor2;
  // return tensor2.DebugString();
}

}  // namespace tensorflow