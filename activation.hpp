#pragma once

/*
	Activation functions, f and it's derivative df
*/
struct Activation
{
	double (*f)(double);
	double (*df)(double);
};
