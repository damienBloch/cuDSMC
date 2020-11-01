from sympy.printing.ccode import C99CodePrinter
class CUDACodePrinter(C99CodePrinter):   
    def _print_Integer(self, expr):
        # we print integers as floats because C code doesn't cast to float for division
        return self._print_Float(expr)
