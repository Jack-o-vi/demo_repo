#include <stdio.h>
#include <conio.h>
#include <time.h>
#include "main.h"

int main()
{
	double st = clock();
	FILE * pFile = fopen("twoo.bmp", "rb");

	// считываем заголовок файла
	BITMAPFILEHEADER header;// __attribute__((unused));

	header.bfType = read_u16(pFile);
	header.bfSize = read_u32(pFile);
	header.bfReserved1 = read_u16(pFile);
	header.bfReserved2 = read_u16(pFile);
	header.bfOffBits = read_u32(pFile);

	// считываем заголовок изображения
	BITMAPINFOHEADER bmiHeader;

	bmiHeader.biSize = read_u32(pFile);
	bmiHeader.biWidth = read_s32(pFile);
	bmiHeader.biHeight = read_s32(pFile);
	bmiHeader.biPlanes = read_u16(pFile);
	bmiHeader.biBitCount = read_u16(pFile);
	bmiHeader.biCompression = read_u32(pFile);
	bmiHeader.biSizeImage = read_u32(pFile);
	bmiHeader.biXPelsPerMeter = read_s32(pFile);
	bmiHeader.biYPelsPerMeter = read_s32(pFile);
	bmiHeader.biClrUsed = read_u32(pFile);
	bmiHeader.biClrImportant = read_u32(pFile);


	RGBQUAD **rgb = new RGBQUAD*[bmiHeader.biWidth];
	for (int i = 0; i < bmiHeader.biWidth; i++) {
		rgb[i] = new RGBQUAD[bmiHeader.biHeight];
	}

	for (int i = 0; i < bmiHeader.biWidth; i++) {
		for (int j = 0; j < bmiHeader.biHeight; j++) {
			rgb[i][j].rgbBlue = getc(pFile);
			rgb[i][j].rgbGreen = getc(pFile);
			rgb[i][j].rgbRed = getc(pFile);
		}

		// пропускаем последний байт в строке
		getc(pFile);
	}

	
	// выводим результат
	for (int i = 0; i < bmiHeader.biWidth; i++) {
		for (int j = 0; j < bmiHeader.biHeight; j++) {
			//	Sleep(500);
			if (rgb[i][j].rgbRed == rgb[i][j].rgbGreen == rgb[i][j].rgbBlue ||
				rgb[i][j].rgbRed == 0 && rgb[i][j].rgbGreen == 0 && rgb[i][j].rgbBlue == 0 	) {
				//printf("%d %d %d ", rgb[i][j].rgbRed, rgb[i][j].rgbGreen, rgb[i][j].rgbBlue);
				printf("%d ", 1);
			} else {
				printf("%d ", 0);
			}
			
		}
		printf("\n");
	}
	double end = clock() - st;
	printf("Time: %f", (end/CLOCKS_PER_SEC));
	fclose(pFile);
	_getch();
	return 0;
}


static unsigned short read_u16(FILE *fp)
{
	unsigned char b0, b1;

	b0 = getc(fp);
	b1 = getc(fp);

	return ((b1 << 8) | b0);
}


static unsigned int read_u32(FILE *fp)
{
	unsigned char b0, b1, b2, b3;

	b0 = getc(fp);
	b1 = getc(fp);
	b2 = getc(fp);
	b3 = getc(fp);

	return ((((((b3 << 8) | b2) << 8) | b1) << 8) | b0);
}


static int read_s32(FILE *fp)
{
	unsigned char b0, b1, b2, b3;

	b0 = getc(fp);
	b1 = getc(fp);
	b2 = getc(fp);
	b3 = getc(fp);

	return ((int)(((((b3 << 8) | b2) << 8) | b1) << 8) | b0);
}