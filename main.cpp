#include "framebuffer.h"
#include "gradient.h"

int main()
{
	const int width = 800, height = 400;
	float* framebuffer = new float[3 * width * height];
	gradientWithCuda(framebuffer, height, width);
	std::string filename = "gradient.ppm";
	render(filename, framebuffer, height, width);
}