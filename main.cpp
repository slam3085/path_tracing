#include "framebuffer.h"
#include "gradient.h"
#include "vec3.h"

int main()
{
    const int width = 800, height = 400;
    vec3* framebuffer = new vec3[width * height];
    gradientWithCuda(framebuffer, height, width);
    std::string filename = "gradient.ppm";
    render(filename, framebuffer, height, width);
    delete[] framebuffer;
}