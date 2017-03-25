// neurol-net-tut.cpp


#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>

using std::cout;
using std::endl;
using std::ofstream;

int main() {

	// Random training sets for XOR -- two inputs and one output
	ofstream fout("trainingData.txt");

	cout << "topology: 2 4 1 " << endl;
	fout << "topology: 2 8 1 " << endl;
	for (int i = 2000; i >= 0; --i) {
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = n1^n2;
		cout << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
		cout << "out: " << t << ".0 " << endl;
		fout << "in: " << n1 << ".0 " << n2 << ".0 " << endl << "out: " << t << ".0 " << endl;

	}

	fout.close();
	system("pause");
	return 0;

}