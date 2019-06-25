#pragma once
struct Point_3 {
	double X;
	double Y;
	double Z;
};
struct Texture {
	double TU;
	double TV;
};
struct NVector {
	double NX;
	double NY;
	double NZ;
};
struct Body {
	int V[3];
	int T[3];
	int N[3];
};