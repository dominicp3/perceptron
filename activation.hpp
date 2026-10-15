#pragma once

/*
	Activation functions, f and its derivative df
*/
struct Activation
{
	double (*f)(double);
	double (*df)(double);
};
