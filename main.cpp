#include "framebuffer.h"
#include "path_tracing.h"
#include "vec3.h"

int main()
{
    const int width = 1280, height = 720;
    vec3* framebuffer = new vec3[width * height];
    path_tracing_with_cuda(framebuffer, height, width);
    std::string filename = "spheres.ppm";
    render(filename, framebuffer, height, width);
    delete[] framebuffer;
}