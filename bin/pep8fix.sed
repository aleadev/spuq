#!/bin/sed -f

# remove blanks after opening braces
s/\([({[]\)[[:blank:]]\+/\1/g

# remove blanks before closing braces
s/[[:blank:]]\+\([])}]\)/\1/g

# remove blanks at the of a line
s/[[:blank:]]\+$//g

# insert blanks after commas
s/,\([^[:blank:]]\)/, \1/g

# insert blank after an operator
s/\([%^<>+*/=-]\+\)\([^[:blank:]]\)/\1 \2/g

# insert blank before an operator
s/\([^[:blank:]]\)\([%^<>+*/=-]\+\)/\1 \2/g

# remove blank again if something like += has been split by the
# previous commands 
s/\([%^<>+*/=-]\+\) =/\1=/g
