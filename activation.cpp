double relu(double x)
{
	if (x > 0.0) {
		return x;
	}

	return 0.0;
}

double dRelu(double x)
{
	if (x > 0.0) {
		return 1.0;
	}

	return 0.0;
}
