float relu(float x)
{
	if (x > 0.0) {
		return x;
	}

	return 0.0;
}

float dRelu(float x)
{
	if (x > 0.0) {
		return 1.0;
	}

	return 0.0;
}
