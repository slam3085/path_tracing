#include <fstream>
#include <string>
#include "vec3.h"


void render(std::string filename, vec3* framebuffer, int height, int width)
{
	std::ofstream file;
	file.open(filename);
	file << "P3\n" << width << ' ' << height << '\n' << "255\n";
	for (int i = height - 1; i >= 0; i--)
		for (int j = 0; j < width; j++)
		{
			int r = int(255.99f * framebuffer[width * i + j].X);
			int g = int(255.99f * framebuffer[width * i + j].Y);
			int b = int(255.99f * framebuffer[width * i + j].Z);
			file << r << ' ' << g << ' ' << b << '\n';
		}
	file.close();
}