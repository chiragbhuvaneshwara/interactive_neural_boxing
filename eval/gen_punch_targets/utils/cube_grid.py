#!/usr/bin/env python3
#
def cube_grid_count(ns):
    # *****************************************************************************80
    #
    ## CUBE_GRID_COUNT counts grid points in a cube (hexahedron, actually).
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    12 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer NS(3), the number of intervals along
    #    each dimension.
    #
    #    Output, integer NG, the number of grid points.
    #
    ng = (ns[0] + 1) * (ns[1] + 1) * (ns[2] + 1)

    return ng


def cube_grid_count_test():
    # *****************************************************************************80
    #
    ## CUBE_GRID_COUNT_TEST tests CUBE_GRID_COUNT.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    12 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('CUBE_GRID_COUNT_TEST:')
    print('  Python version: %s' % (platform.python_version()))
    print('  CUBE_GRID_COUNT can count the points in a grid')
    print('  over a cube (hexahedron) with NS(I)+1 grid points')
    print('  in the I-th direction.')

    ns = np.zeros(3)

    print('')
    print('          NS                 NG')
    print('  --------------------')
    print('')

    for i in [1, 2, 4, 8]:
        ns[0] = i
        for j in [2, 3]:
            ns[1] = j
            for k in [4, 10]:
                ns[2] = k
                ng = cube_grid_count(ns)
                print('  ( %4d, %4d, %4d ) %8d' % (ns[0], ns[1], ns[2], ng))
    #
    #  Terminate.
    #
    print('')
    print('CUBE_GRID_COUNT_TEST:')
    print('  Normal end of execution.')
    return


def cube_grid_display(xv, ng, xg, filename):
    # *****************************************************************************80
    #
    ## CUBE_GRID_DISPLAY displays grid points inside a cube (hexahedron).
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    13 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, real XV[8,3], the vertices of the hexahedron, in a particular order.
    #
    #    Input, integer NG, the number of grid points.
    #
    #    Input, real XG[NG,3], the grid points.
    #
    #    Input, string FILENAME, the name of the plotfile to be created.
    #
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    #  Draw the grid points.
    #
    ax.scatter(xg[:, 0], xg[:, 1], xg[:, 2], 'b')
    #
    #  Outline the hexahedron by its edges.
    #
    ax.plot([xv[0, 0], xv[1, 0]], [xv[0, 1], xv[1, 1]], [xv[0, 2], xv[1, 2]], 'r')
    ax.plot([xv[0, 0], xv[2, 0]], [xv[0, 1], xv[2, 1]], [xv[0, 2], xv[2, 2]], 'r')
    ax.plot([xv[0, 0], xv[4, 0]], [xv[0, 1], xv[4, 1]], [xv[0, 2], xv[4, 2]], 'r')
    ax.plot([xv[1, 0], xv[3, 0]], [xv[1, 1], xv[3, 1]], [xv[1, 2], xv[3, 2]], 'r')
    ax.plot([xv[1, 0], xv[5, 0]], [xv[1, 1], xv[5, 1]], [xv[1, 2], xv[5, 2]], 'r')
    ax.plot([xv[2, 0], xv[3, 0]], [xv[2, 1], xv[3, 1]], [xv[2, 2], xv[3, 2]], 'r')
    ax.plot([xv[2, 0], xv[6, 0]], [xv[2, 1], xv[6, 1]], [xv[2, 2], xv[6, 2]], 'r')
    ax.plot([xv[3, 0], xv[7, 0]], [xv[3, 1], xv[7, 1]], [xv[3, 2], xv[7, 2]], 'r')
    ax.plot([xv[4, 0], xv[5, 0]], [xv[4, 1], xv[5, 1]], [xv[4, 2], xv[5, 2]], 'r')
    ax.plot([xv[4, 0], xv[6, 0]], [xv[4, 1], xv[6, 1]], [xv[4, 2], xv[6, 2]], 'r')
    ax.plot([xv[5, 0], xv[7, 0]], [xv[5, 1], xv[7, 1]], [xv[5, 2], xv[7, 2]], 'r')
    ax.plot([xv[6, 0], xv[7, 0]], [xv[6, 1], xv[7, 1]], [xv[6, 2], xv[7, 2]], 'r')

    ax.set_xlabel('<---X--->')
    ax.set_ylabel('<---Y--->')
    ax.set_zlabel('<---Z--->')
    ax.set_title('Grid points in hexahedron')
    ax.grid(True)
    # ax.axis ( 'equal' )
    plt.savefig(filename)
    # plt.show ( )

    plt.clf()

    return


def cube_grid_display_test():
    # *****************************************************************************80
    #
    ## CUBE_GRID_DISPLAY displays grid points inside a cube (hexahedron).
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    13 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('CUBE_GRID_DISPLAY_TEST:')
    print('  Python version: %s' % (platform.python_version()))
    print('  CUBE_GRID_DISPLAY can display a grid of points in a cube.')

    xv = np.array([ \
        [0.0, 0.0, 5.0], \
        [1.0, 0.0, 5.0], \
        [0.0, 1.0, 5.0], \
        [1.0, 1.0, 5.0], \
        [0.0, 2.0, 8.0], \
        [1.0, 2.0, 8.0], \
        [0.0, 4.0, 8.0], \
        [1.0, 4.0, 8.0]])

    ng = 16
    xg = np.array([ \
        [0.0, 0.0, 5.0], \
        [0.0, 1.0, 5.0], \
        [1.0, 0.0, 5.0], \
        [1.0, 1.0, 5.0], \
        [0.0, 0.66, 6.0], \
        [0.0, 2.0, 6.0], \
        [1.0, 0.66, 6.0], \
        [1.0, 2.0, 6.0], \
        [0.0, 1.33, 7.0], \
        [0.0, 3.0, 7.0], \
        [1.0, 1.33, 7.0], \
        [1.0, 3.0, 7.0], \
        [0.0, 2.0, 8.0], \
        [0.0, 4.0, 8.0], \
        [1.0, 2.0, 8.0], \
        [1.0, 4.0, 8.0]])

    filename = 'cube_grid_display.png'

    cube_grid_display(xv, ng, xg, filename)
    #
    #  Terminate.
    #
    print('')
    print('CUBE_GRID_DISPLAY_TEST:')
    print('  Normal end of execution.')
    return


def cube_grid_points(ns, xv, ng):
    # *****************************************************************************80
    #
    ## CUBE_GRID_POINTS: grid points over a quadrilateral.
    #
    #  Discussion:
    #
    #    The 8 vertices of the bounding quadrilateral should be given in the
    #    the order suggested by the following list:
    #
    #      000
    #      X00
    #      0Y0
    #      XY0
    #      00Z
    #      X0Z
    #      0YZ
    #      XYZ
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    13 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer NS[3], the number of points along each dimension.
    #
    #    Input, real XV[8,3], the vertices, in the order
    #
    #    Input, integer NG, the number of points.
    #    NG = NS[0] * NS[1] * NS[3].
    #
    #    Output, real XG[NG,3], the grid points.
    #
    import numpy as np
    #
    #  Create the 1D grids in each dimension.
    #
    l = ns[0]
    m = ns[1]
    n = ns[2]

    ng = (l + 1) * (m + 1) * (n + 1)
    xg = np.zeros((ng, 3))

    ig = 0
    for k in range(0, ns[2] + 1):
        for j in range(0, ns[1] + 1):
            for i in range(0, ns[0] + 1):
                a000 = float((l - i) * (m - j) * (n - k)) / float(l * m * n)
                ax00 = float((i) * (m - j) * (n - k)) / float(l * m * n)
                a0y0 = float((l - i) * (j) * (n - k)) / float(l * m * n)
                axy0 = float((i) * (j) * (n - k)) / float(l * m * n)
                a00z = float((l - i) * (m - j) * (k)) / float(l * m * n)
                ax0z = float((i) * (m - j) * (k)) / float(l * m * n)
                a0yz = float((l - i) * (j) * (k)) / float(l * m * n)
                axyz = float((i) * (j) * (k)) / float(l * m * n)

                xg[ig, 0] = a000 * xv[0, 0] + ax00 * xv[1, 0] \
                            + a0y0 * xv[2, 0] + axy0 * xv[3, 0] \
                            + a00z * xv[4, 0] + ax0z * xv[5, 0] \
                            + a0yz * xv[6, 0] + axyz * xv[7, 0]

                xg[ig, 1] = a000 * xv[0, 1] + ax00 * xv[1, 1] \
                            + a0y0 * xv[2, 1] + axy0 * xv[3, 1] \
                            + a00z * xv[4, 1] + ax0z * xv[5, 1] \
                            + a0yz * xv[6, 1] + axyz * xv[7, 1]

                xg[ig, 2] = a000 * xv[0, 2] + ax00 * xv[1, 2] \
                            + a0y0 * xv[2, 2] + axy0 * xv[3, 2] \
                            + a00z * xv[4, 2] + ax0z * xv[5, 2] \
                            + a0yz * xv[6, 2] + axyz * xv[7, 2]
                ig = ig + 1

    return xg


def cube_grid_points_test(l, m, n):
    # *****************************************************************************80
    #
    ## CUBE_GRID_POINTS_TEST tests CUBE_GRID_POINTS.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    13 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('CUBE_GRID_POINTS_TEST:')
    print('  Python version: %s' % (platform.python_version()))
    print('  CUBE_GRID_POINTS defines a grid')
    print('  with (L,M,N) points on a side.')

    print('')
    print('  Input value of L is %d' % (l))
    print('  Input value of M is %d' % (m))
    print('  Input value of N is %d' % (n))

    ns = np.zeros(3, dtype=np.int32);
    ns[0] = l
    ns[1] = m
    ns[2] = n

    ng = cube_grid_count(ns)
    print('')
    print('  Number of grid points will be %d' % (ng))

    xv = np.array([ \
        [-0.4, 1.05, 0.45], \
        [0.1, 1.05, 0.45], \
        [-0.4, 1.75, 0.45], \
        [0.1, 1.75, 0.45], \
        [-0.4, 1.05, 0.78], \
        [0.1, 1.05, 0.78], \
        [-0.4, 1.75, 0.78], \
        [0.1, 1.75, 0.78]])

    r83col_print_part(8, xv, 20, '  Hexahedron vertices:')

    xg = cube_grid_points(ns, xv, ng)

    r83col_print_part(ng, xg, 20, '  Part of the grid point array:');
    #
    #  Write the data to a file.
    #
    filename = 'cube_grid_points.xyz'
    r8mat_write(filename, ng, 3, xg)
    print('')
    print('  Data written to the file "%s".' % (filename))
    #
    #  Plot the data.
    #
    filename = 'cube_grid_points.png'
    cube_grid_display(xv, ng, xg, filename)
    print('')
    print('  Plot written to the file "%s".' % (filename))
    #
    #  Terminate.
    #
    print('')
    print('CUBE_GRID_POINTS_TEST:')
    print('  Normal end of execution.')
    return


def cube_grid_test():
    # *****************************************************************************80
    #
    ## CUBE_GRID_TEST tests the CUBEE_GRID library.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    13 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    print('')
    print('CUBE_GRID_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  Test the CUBE_GRID library.')
    #
    #  Utilities:
    #
    r83col_print_part_test()
    r8mat_write_test()
    timestamp_test()
    #
    #  Library.
    #
    cube_grid_display_test()
    cube_grid_count_test()
    cube_grid_points_test(3, 2, 2)
    #
    #  Terminate.
    #
    print('')
    print('CUBE_GRID_TEST:')
    print('  Normal end of execution.')
    return


def r83col_print_part(n, a, max_print, title):
    # *****************************************************************************80
    #
    ## R83COL_PRINT_PART prints "part" of an R83COL.
    #
    #  Discussion:
    #
    #    An R83COL is a (3,N) array of R8's.
    #
    #    The user specifies MAX_PRINT, the maximum number of lines to print.
    #
    #    If N, the size of the vector, is no more than MAX_PRINT, then
    #    the entire vector is printed, one entry per line.
    #
    #    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
    #    followed by a line of periods suggesting an omission,
    #    and the last entry.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    11 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer N, the number of entries of the vector.
    #
    #    Input, real A(N,3), the vector to be printed.
    #
    #    Input, integer MAX_PRINT, the maximum number of lines
    #    to print.
    #
    #    Input, string TITLE, a title.
    #
    if (0 < max_print):

        if (0 < n):

            if (0 < len(title)):
                print('')
                print(title)

            print('')

            if (n <= max_print):

                for i in range(0, n):
                    print('  %4d  %14g  %14g  %14g' % (i, a[i, 0], a[i, 1], a[i, 2]))

            elif (3 <= max_print):

                for i in range(0, max_print - 2):
                    print('  %4d  %14g  %14g  %14g' % (i, a[i, 0], a[i, 1], a[i, 2]))
                print('  ....  ..............  ..............  ..............')
                i = n - 1
                print('  %4d  %14g  %14g  %14g' % (i, a[i, 0], a[i, 1], a[i, 2]))

            else:

                for i in range(0, max_print - 1):
                    print('  %4d  %14g  %14g  %14g' % (i, a[i, 0], a[i, 1], a[i, 2]))
                i = max_print - 1
                print('  %4d  %14g  %14g  %14g  ...more entries...' \
                      % (i, a[i, 0], a[i, 1], a[i, 2]))

    return


def r83col_print_part_test():
    # *****************************************************************************80
    #
    ## R83COL_PRINT_PART_TEST tests R83COL_PRINT_PART.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    11 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('R83COL_PRINT_PART_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  R83COL_PRINT_PART prints part of an R83COL.')

    n = 10

    v = np.array([ \
        [11, 12, 13], \
        [21, 22, 23], \
        [31, 32, 33], \
        [41, 42, 43], \
        [51, 52, 53], \
        [61, 62, 63], \
        [71, 72, 73], \
        [81, 82, 83], \
        [91, 92, 93], \
        [101, 102, 103]])

    max_print = 2
    r83col_print_part(n, v, max_print, '  Output with MAX_PRINT = 2')

    max_print = 5
    r83col_print_part(n, v, max_print, '  Output with MAX_PRINT = 5')

    max_print = 25
    r83col_print_part(n, v, max_print, '  Output with MAX_PRINT = 25')
    #
    #  Terminate.
    #
    print('')
    print('R83COL_PRINT_PART_TEST:')
    print('  Normal end of execution.')
    return


def r8mat_write(filename, m, n, a):
    # *****************************************************************************80
    #
    ## R8MAT_WRITE writes an R8MAT to a file.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    12 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, string FILENAME, the name of the output file.
    #
    #    Input, integer M, the number of rows in A.
    #
    #    Input, integer N, the number of columns in A.
    #
    #    Input, real A(M,N), the matrix.
    #
    output = open(filename, 'w')

    for i in range(0, m):
        for j in range(0, n):
            s = '  %g' % (a[i, j])
            output.write(s)
        output.write('\n')

    output.close()

    return


def r8mat_write_test():
    # *****************************************************************************80
    #
    ## R8MAT_WRITE_TEST tests R8MAT_WRITE.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    12 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('R8MAT_WRITE_TEST:')
    print('  Python version: %s' % (platform.python_version()))
    print('  Test R8MAT_WRITE, which writes an R8MAT to a file.')

    filename = 'r8mat_write.txt'
    m = 5
    n = 3
    a = np.array(( \
        (1.1, 1.2, 1.3), \
        (2.1, 2.2, 2.3), \
        (3.1, 3.2, 3.3), \
        (4.1, 4.2, 4.3), \
        (5.1, 5.2, 5.3)))
    r8mat_write(filename, m, n, a)

    print('')
    print('  Created file "%s".' % (filename))
    #
    #  Terminate.
    #
    print('')
    print('R8MAT_WRITE_TEST:')
    print('  Normal end of execution.')
    return


def timestamp():
    # *****************************************************************************80
    #
    ## TIMESTAMP prints the date as a timestamp.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 April 2013
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    None
    #
    import time

    t = time.time()
    print(time.ctime(t))

    return None


def timestamp_test():
    # *****************************************************************************80
    #
    ## TIMESTAMP_TEST tests TIMESTAMP.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    03 December 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    None
    #
    import platform

    print('')
    print('TIMESTAMP_TEST:')
    print('  Python version: %s' % (platform.python_version()))
    print('  TIMESTAMP prints a timestamp of the current date and time.')
    print('')

    timestamp()
    #
    #  Terminate.
    #
    print('')
    print('TIMESTAMP_TEST:')
    print('  Normal end of execution.')
    return


if (__name__ == '__main__'):
    timestamp()
    cube_grid_test()
    timestamp()