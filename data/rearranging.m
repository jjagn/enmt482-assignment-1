syms z x a b c

eqn1 = z == 1/(x + a) + b * x + c
eqn2 = z == log(a * x) / (x^2) + b

solve(eqn1, x)
solve(eqn2, x, 'IgnoreAnalyticConstraints',true)