#include <fstream>
#include <string>


void render(std::string filename, float* framebuffer, int height, int width)
{
	std::ofstream file;
	file.open("something.ppm");
	file << "P3\n" << width << ' ' << height << '\n' << "255\n";
	for (int i = height - 1; i >= 0; i--)
		for (int j = 0; j < width; j++)
		{
			int r = int(255.99 * framebuffer[3 * (width * i + j) + 0]);
			int g = int(255.99 * framebuffer[3 * (width * i + j) + 1]);
			int b = int(255.99 * framebuffer[3 * (width * i + j) + 2]);
			file << r << ' ' << g << ' ' << b << '\n';
		}
	file.close();
}