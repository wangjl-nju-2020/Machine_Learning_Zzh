def print_formatted_vec(vec, n):
    print('(', end='')
    for i in range(n - 1):
        print('%.5f, ' % vec[i], end='')
    print('%.5f)' % vec[-1])

