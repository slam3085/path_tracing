#include "path_tracing.h"
#include "vec3.h"
#include <string>

int main()
{
    const int width = 1024, height = 1024;
    std::string filename = "boxes.png";
    path_tracing_with_cuda(filename, height, width);
}