#include "path_tracing.h"
#include "vec3.h"
#include <string>

int main()
{
    const int width = 1280, height = 720;
    std::string filename = "spheres.ppm";
    path_tracing_with_cuda(filename, height, width);
}