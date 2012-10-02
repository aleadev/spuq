import pdb
class Mdb(pdb.Pdb):
    #def complete_print(self, text, line, begidx, endidx):
    #    return complete_p(self, text, line, begidx, endidx)

    def complete_p(self, text, line, begidx, endidx):
        info = ""
        try:
            if "." in text:
                expr, match = text.rsplit(".", 1)
                info = expr
                val = eval(expr, self.curframe.f_globals,
                           self.curframe_locals)
                choice = dir(val)
                pref = expr + "."
            else:
                choice = self.curframe_locals.keys()
                match  = text
                pref = ""
            vals = [pref + val for val in choice if val.startswith(match)]
            if len(vals)==1 and vals[0]==text:
                #print "FOOO", text, vals[0], ">>>"
                t = eval(text, self.curframe.f_globals, self.curframe_locals)
                if callable(t): 
                    vals = [vals[0] + "()"]
        except Exception as e:
            print "Exc: (%r, %r)" % (e, info)
            return []

        return vals


    #compfunc(text, line, begidx, endidx)
