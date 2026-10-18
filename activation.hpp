#pragma once

/*
	Activation functions, f and its derivative df
*/
struct Activation
{
	float (*f)(float);
	float (*df)(float);
};

float relu(float x);
float dRelu(float x);
