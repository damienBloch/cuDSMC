from sympy.printing.ccode import C99CodePrinter
class codePrinter(C99CodePrinter):   
    def _print_Integer(self, expr):
        return self._print_Float(expr)
