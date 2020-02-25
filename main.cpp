#include "framebuffer.h"
#include "path_tracing.h"
#include "vec3.h"

int main()
{
    const int width = 1600, height = 800;
    vec3* framebuffer = new vec3[width * height];
    path_tracing_with_cuda(framebuffer, height, width);
    std::string filename = "two_spheres.ppm";
    render(filename, framebuffer, height, width);
    delete[] framebuffer;
}