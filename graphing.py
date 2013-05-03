"""graphing.py -- helper functions for graphing with matplotlib/pyplot

This software is licensed under the terms of the MIT License as
follows:

Copyright (c) 2013 Jessica B. Hamrick

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', size=12)


def save(path, fignum=None, close=True, width=None, height=None,
         ext=None, verbose=False):
    """Save a figure from pyplot.

    Parameters:

    path [string] : The path (and filename, without the extension) to
    save the figure to.

    fignum [integer] : The id of the figure to save. If None, saves the
    current figure.

    close [boolean] : (default=True) Whether to close the figure after
    saving.  If you want to save the figure multiple times (e.g., to
    multiple formats), you should NOT close it in between saves or you
    will have to re-plot it.

    width [number] : The width that the figure should be saved with. If
    None, the current width is used.

    height [number] : The height that the figure should be saved with. If
    None, the current height is used.

    ext [string or list of strings] : (default='png') The file
    extension. This must be supported by the active matplotlib backend
    (see matplotlib.backends module).  Most backends support 'png',
    'pdf', 'ps', 'eps', and 'svg'.

    verbose [boolean] : (default=True) Whether to print information
    about when and where the image has been saved.

    """

    # get the figure
    if fignum is not None:
        fig = plt.figure(fignum)
    else:
        fig = plt.gcf()

    # set its dimenions
    if width:
        fig.set_figwidth(width)
    if height:
        fig.set_figheight(height)

    # make sure we have a list of extensions
    if ext is not None and not hasattr(ext, '__iter__'):
        ext = [ext]

    # Extract the directory and filename from the given path
    directory, basename = os.path.split(path)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # infer the extension if ext is None
    if ext is None:
        basename, ex = os.path.splitext(basename)
        ext = [ex[1:]]

    for ex in ext:
        # The final path to save to
        filename = "%s.%s" % (basename, ex)
        savepath = os.path.join(directory, filename)

        if verbose:
            sys.stdout.write("Saving figure to '%s'..." % savepath)

        # Actually save the figure
        plt.savefig(savepath)

    # Close it
    if close:
        plt.close()

    if verbose:
        sys.stdout.write("Done\n")


def plot_to_array(fig=None):
    """Convert a matplotlib figure `fig` to RGB pixels.

    Note that this DOES include things like ticks, labels, title, etc.

    Parameters
    ----------
    fig : int or matplotlib.figure.Figure (default=None)
        A matplotlib figure or figure identifier. If None, the current
        figure is used.

    Returns
    -------
    out : numpy.ndarray
        (width, height, 4) array of RGB values

    """

    # get the figure object
    if fig is None:
        fig = plt.gcf()
    try:
        fig = int(fig)
    except TypeError:
        pass
    else:
        fig = plt.figure(fig)

    # render the figure
    fig.canvas.draw()

    # convert the figure to a rgb string, and read that buffer into a
    # numpy array
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.renderer.tostring_rgb()
    arr = np.fromstring(buf, dtype=np.uint8)
    arr.resize(h, w, 3)

    # convert values from 0-255 to 0-1
    farr = arr / 255.

    return farr
