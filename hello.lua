function fact(n)
	if n < 2 then
		return 1
	else 
		return n * fact()
