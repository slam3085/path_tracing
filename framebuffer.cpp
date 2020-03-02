#include <fstream>
#include <string>
#include <cmath>
#include "vec3.h"


void render(std::string filename, vec3* framebuffer, int height, int width, int rays_per_pixel)
{
	std::ofstream file;
	file.open(filename);
	file << "P3\n" << width << ' ' << height << '\n' << "255\n";
	for (int i = height - 1; i >= 0; i--)
		for (int j = 0; j < width; j++)
		{
			int r = int(255.99f * sqrt(framebuffer[width * i + j].X / float(rays_per_pixel)));
			int g = int(255.99f * sqrt(framebuffer[width * i + j].Y / float(rays_per_pixel)));
			int b = int(255.99f * sqrt(framebuffer[width * i + j].Z / float(rays_per_pixel)));
			file << r << ' ' << g << ' ' << b << '\n';
		}
	file.close();
}