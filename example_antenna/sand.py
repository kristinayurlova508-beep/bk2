from sympy import symbols, sin, exp, sqrt, pi
x, y = symbols("x y", real=True)
amplitude = 1.0
lambda_val = 0.3
x0, y0 = 1.0, 1.0
absorption = 1.8

r = sqrt((x-x0)**2 + (y-y0)**2)
u_expr = amplitude*sin(2*pi*r/lambda_val)*exp(-absorption*r)
