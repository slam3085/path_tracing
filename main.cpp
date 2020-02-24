#include "framebuffer.h"
#include "ray_pathing.h"
#include "vec3.h"

int main()
{
	const int width = 800, height = 400;
    vec3* framebuffer = new vec3[width * height];
	ray_pathing_with_cuda(framebuffer, height, width);
	std::string filename = "sphere.ppm";
	render(filename, framebuffer, height, width);
    delete[] framebuffer;
}