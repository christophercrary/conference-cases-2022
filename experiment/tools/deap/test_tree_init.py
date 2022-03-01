def generate_program(pset, d_max, s_desired):
    """Attempt to generate a program of the specified size.
    
    The program is generated based on the given primitive set 
    and maximum depth constraints. Different tree shapes are
    randomly sampled based on the `terminalRatio` attribute
    of the primitive set.

    It is not guaranteed that a program of the specified size
    will be found, even if one exists in theory.

    Keyword arguments:
    pset -- Primitive set, of type `deap.gp.PrimitiveSet`.
    d_max -- Maximum allowable depth of program.
    s_desired -- Desired size of program.
    """
    program = []  # Prefix representation of program.
    size = 1      # Size of program.
    stack = [0]   # Stack of depths for nodes outstanding.

    while len(stack) != 0:
        d = stack.pop()  # Depth of the next relevant node.
        # Functions with an arity that is not too large.
        fn = [f for f in pset.primitives[pset.ret] if size+f.arity <= s_desired]
        # Maximum arity of the above functions.
        a_max = 0 if fn == [] else max([f.arity for f in fn])

        # Determine which of the above functions have an arity 
        # that is not too small.
        temp = []
        for f in fn:
            # If the current node is chosen to be `f`...
            # Maximum possible size of the subprogram rooted at node.
            s = f.arity * get_max_size(a_max, d_max-(d+1))
            # Maximum possible size of the overall program.
            s_max = (size + s if stack == [] else size + s + sum(
                [get_max_size(a_max, d_max-d)-1 for d in stack]))
            if s_max >= s_desired:
                temp.append(f)  # Allowable function.

        # Functions with an arity that is not too small.
        fn = temp
        a_max = 0 if fn == [] else max([f.arity for f in fn])

        if fn == [] and size < s_desired:
            return None  # Invalid program.

        # Maximum possible program size if the node currently
        # under consideration is chosen to be a terminal.
        s_max = (size if stack == [] or fn == [] else 
            size + sum([get_max_size(a_max, d_max-d)-1 for d in stack]))

        # Boolean to determine if the current node should be a terminal.
        choose_terminal = (
            (fn == []) or (d == d_max) or (size == s_desired) or (
            random.random() < pset.terminalRatio and s_max >= s_desired))

        if choose_terminal:
            
            # A random terminal node is to be chosen.
            terminal = random.choice(pset.terminals[pset.ret])
            terminal = terminal() if isclass(terminal) else terminal
            program.append(terminal)
        else:
            # A random (valid) function node is to be chosen.
            f = random.choice(fn)
            program.append(f)
            for _ in range(f.arity):
                stack.append(d + 1)  # Add children.
            size += f.arity  # Update program size.
    # Program was generated.
    if size != s_desired:
        return None  # Invalid program.
    else:
        return program  # Valid program.