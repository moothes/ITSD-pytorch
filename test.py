import src

def test():

	L = src.Loader('test')
	E = src.Eval(L)
	TE = src.Tester(L, E)
	TE.test()

if __name__ == '__main__':
	
	test()