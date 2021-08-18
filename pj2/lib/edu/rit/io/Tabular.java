//******************************************************************************
//
// File:    Tabular.java
// Package: edu.rit.io
// Unit:    Class edu.rit.io.Tabular
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.io;

import edu.rit.util.AList;
import edu.rit.util.IntList;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;

/**
 * Class Tabular provides an object for formatting output in a table of rows and
 * columns. To set up a table:
 * <OL TYPE=1>
 * <P><LI>
 * Construct a new instance of class Tabular, specifying the initial number of
 * rows and columns (which may be 0).
 * <P><LI>
 * Call the <TT>addRow()</TT>, <TT>addRows()</TT>, <TT>addColumn()</TT>, or
 * <TT>addColumns()</TT> methods to add rows or columns at any time. Rows are
 * added at the bottom. Columns are added at the right side.
 * <P><LI>
 * Call the <TT>columnWidth()</TT> method to set the width of a column. The
 * default is "automatic," where the column's width is the maximum length of the
 * text in any cell in the column. If a column has a specific width, any cell
 * text beyond that width is not printed.
 * <P><LI>
 * Call the <TT>gap()</TT> method to set the width of the gap between adjacent
 * columns. The same gap is used for all columns. The default gap is 2.
 * <P><LI>
 * Call the <TT>justify()</TT> method to set left, center, or right
 * justification for all cells in a column. The default is left justification.
 * <P><LI>
 * Call the <TT>clear()</TT>, <TT>set()</TT>, <TT>append()</TT>,
 * <TT>setf()</TT>, or <TT>appendf()</TT> methods to clear, set, or append to
 * the text in a cell.
 * <P><LI>
 * Call one of the <TT>print()</TT> methods to print the table on a PrintStream
 * or a PrintWriter.
 * </OL>
 *
 * @author  Alan Kaminsky
 * @version 12-May-2016
 */
public class Tabular
	{

// Exported constants.

	/**
	 * Automatic column width.
	 */
	public static final int AUTOMATIC = -1;

	/**
	 * Left justification.
	 */
	public static final int LEFT = 0;

	/**
	 * Center justification.
	 */
	public static final int CENTER = 1;

	/**
	 * Right justification.
	 */
	public static final int RIGHT = 2;

// Hidden data members.

	// Number of rows and columns.
	private int R;
	private int C;

	// List of lists of cells. First index = row. Second index = column.
	private AList<AList<String>> cell = new AList<AList<String>>();

	// List of column widths.
	private IntList width = new IntList();

	// List of column justifications.
	private IntList justification = new IntList();

	// Inter-column gap.
	private int gap = 2;

// Exported constructors.

	/**
	 * Construct a new table with no rows and no columns. Rows and columns must
	 * be added later.
	 */
	public Tabular()
		{
		}

	/**
	 * Construct a new table with the given number of rows and columns. Rows and
	 * columns may be added later.
	 *
	 * @param  R  Number of rows.
	 * @param  C  Number of columns.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>R</TT> &lt; 1. Thrown if
	 *     <TT>C</TT> &lt; 1.
	 */
	public Tabular
		(int R,
		 int C)
		{
		addRows (R);
		addColumns (C);
		}

// Exported operations.

	/**
	 * Add one row at the bottom of this table.
	 *
	 * @return  Index of row that was added.
	 */
	public int addRow()
		{
		return addRows (1);
		}

	/**
	 * Add the given number of rows at the bottom of this table.
	 *
	 * @param  R  Number of rows.
	 *
	 * @return  Index of first row that was added.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>R</TT> &lt; 1.
	 */
	public int addRows
		(int R)
		{
		if (R < 1)
			throw new IllegalArgumentException (String.format
				("Tabular.addRows(): R = %d illegal", R));
		for (int i = 0; i < R; ++ i)
			{
			AList<String> row = cell.addLast (new AList<String>());
			for (int j = 0; j < C; ++ j)
				row.addLast ("");
			}
		this.R += R;
		return this.R - R;
		}

	/**
	 * Add one column at the right side of this table.
	 *
	 * @return  Index of column that was added.
	 */
	public int addColumn()
		{
		return addColumns (1);
		}

	/**
	 * Add the given number of columns at the right side of this table.
	 *
	 * @param  C  Number of columns.
	 *
	 * @return  Index of first column that was added.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>C</TT> &lt; 1.
	 */
	public int addColumns
		(int C)
		{
		if (C < 1)
			throw new IllegalArgumentException (String.format
				("Tabular.addColumns(): C = %d illegal", C));
		for (int i = 0; i < C; ++ i)
			{
			for (int j = 0; j < R; ++ j)
				cell.get(j).addLast ("");
			width.addLast (AUTOMATIC);
			justification.addLast (LEFT);
			}
		this.C += C;
		return this.C - C;
		}

	/**
	 * Set the width of the given column in this table.
	 *
	 * @param  c  Column index, 0 .. (number of columns)&minus;1.
	 * @param  w  Column width, or {@link #AUTOMATIC}.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>c</TT> is out of bounds.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>w</TT> &lt; 0 and <TT>w</TT> &ne;
	 *     {@link #AUTOMATIC}.
	 */
	public void columnWidth
		(int c,
		 int w)
		{
		if (w < AUTOMATIC)
			throw new IllegalArgumentException (String.format
				("Tabular.columnWidth(): w = %d illegal", w));
		width.set (c, w);
		}

	/**
	 * Set the gap between adjacent columns in this table.
	 *
	 * @param  g  Gap.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>g</TT> &lt; 0.
	 */
	public void gap
		(int g)
		{
		if (g < 0)
			throw new IllegalArgumentException (String.format
				("Tabular.gap(): g = %d illegal", g));
		this.gap = g;
		}

	/**
	 * Set the justification of the given column in this table.
	 *
	 * @param  c  Column index, 0 .. (number of columns)&minus;1.
	 * @param  j  Justification: {@link #LEFT}, {@link #CENTER}, or
	 *            {@link #RIGHT}.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>c</TT> is out of bounds.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>j</TT> is not one of the above.
	 */
	public void justify
		(int c,
		 int j)
		{
		if (LEFT > j || j > RIGHT)
			throw new IllegalArgumentException (String.format
				("Tabular.justify(): j = %d illegal", j));
		justification.set (c, j);
		}

	/**
	 * Clear the text in the given cell in this table.
	 *
	 * @param  r  Cell row, 0 .. (number of rows)&minus;1.
	 * @param  c  Cell column, 0 .. (number of columns)&minus;1.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>r</TT> or <TT>c</TT> is out of
	 *     bounds.
	 */
	public void clear
		(int r,
		 int c)
		{
		cell.get(r).set (c, "");
		}

	/**
	 * Set the given text in the given cell in this table.
	 *
	 * @param  r  Cell row, 0 .. (number of rows)&minus;1.
	 * @param  c  Cell column, 0 .. (number of columns)&minus;1.
	 * @param  t  Cell text. If null, the cell is cleared.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>r</TT> or <TT>c</TT> is out of
	 *     bounds.
	 */
	public void set
		(int r,
		 int c,
		 String t)
		{
		cell.get(r).set (c, t == null ? "" : t);
		}

	/**
	 * Append the given text to the given cell in this table.
	 *
	 * @param  r  Cell row, 0 .. (number of rows)&minus;1.
	 * @param  c  Cell column, 0 .. (number of columns)&minus;1.
	 * @param  t  Text to append. If null, the cell is not altered.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>r</TT> or <TT>c</TT> is out of
	 *     bounds.
	 */
	public void append
		(int r,
		 int c,
		 String t)
		{
		if (t != null)
			{
			AList<String> row = cell.get (r);
			row.set (c, row.get(c) + t);
			}
		}

	/**
	 * Format the given objects and set the text in the given cell in this
	 * table.
	 *
	 * @param  r    Cell row, 0 .. (number of rows)&minus;1.
	 * @param  c    Cell column, 0 .. (number of columns)&minus;1.
	 * @param  fmt  Format string (see class {@linkplain java.util.Formatter
	 *              java.util.Formatter}).
	 * @param  obj  Objects to format.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>r</TT> or <TT>c</TT> is out of
	 *     bounds.
	 */
	public void setf
		(int r,
		 int c,
		 String fmt,
		 Object... obj)
		{
		set (r, c, String.format (fmt, obj));
		}

	/**
	 * Format the given objects and append to the text in the given cell in this
	 * table.
	 *
	 * @param  r    Cell row, 0 .. (number of rows)&minus;1.
	 * @param  c    Cell column, 0 .. (number of columns)&minus;1.
	 * @param  fmt  Format string (see class {@linkplain java.util.Formatter
	 *              java.util.Formatter}).
	 * @param  obj  Objects to format.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>r</TT> or <TT>c</TT> is out of
	 *     bounds.
	 */
	public void appendf
		(int r,
		 int c,
		 String fmt,
		 Object... obj)
		{
		append (r, c, String.format (fmt, obj));
		}

	/**
	 * Print this table on <TT>System.out</TT>.
	 */
	public void print()
		{
		print (new PrintableStream (System.out));
		}

	/**
	 * Print this table on the given print stream.
	 *
	 * @param  out  Print stream.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 */
	public void print
		(PrintStream out)
		{
		print (new PrintableStream (out));
		}

	/**
	 * Print this table on the given print writer.
	 *
	 * @param  out  Print writer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 */
	public void print
		(PrintWriter out)
		{
		print (new PrintableWriter (out));
		}

// Hidden helper classes and interfaces.

	/**
	 * Interface for an object that can print strings and characters.
	 */
	private static interface Printable
		{
		public void print (String s);
		public void print (char c);
		public void println();
		public void flush();
		}

	/**
	 * Printable class that wraps a PrintStream.
	 */
	private static class PrintableStream
		implements Printable
		{
		private PrintStream out;
		public PrintableStream (PrintStream out)
			{
			this.out = out;
			}
		public void print (String s)
			{
			out.print (s);
			}
		public void print (char c)
			{
			out.print (c);
			}
		public void println()
			{
			out.println();
			}
		public void flush()
			{
			out.flush();
			}
		}

	/**
	 * Printable class that wraps a PrintWriter.
	 */
	private static class PrintableWriter
		implements Printable
		{
		private PrintWriter out;
		public PrintableWriter (PrintWriter out)
			{
			this.out = out;
			}
		public void print (String s)
			{
			out.print (s);
			}
		public void print (char c)
			{
			out.print (c);
			}
		public void println()
			{
			out.println();
			}
		public void flush()
			{
			out.flush();
			}
		}

// Hidden operations.

	/**
	 * Print this table on the given printable object.
	 *
	 * @param  out  Printable object.
	 */
	private void print
		(Printable out)
		{
		// Determine actual width of each column.
		int[] colwidth = new int [C];
		for (int c = 0; c < C; ++ c)
			{
			int w = width.get (c);
			if (w == AUTOMATIC)
				{
				w = 0;
				for (int r = 0; r < R; ++ r)
					w = Math.max (w, cell.get(r).get(c).length());
				}
			colwidth[c] = w;
			}

		// Print cell contents.
		for (int r = 0; r < R; ++ r)
			{
			AList<String> row = cell.get (r);
			for (int c = 0; c < C; ++ c)
				printCell (out, colwidth[c], justification.get(c),
					c == 0 ? 0 : gap, c == C - 1, row.get (c));
			out.println();
			out.flush();
			}
		}

	/**
	 * Print the given cell's contents.
	 *
	 * @param  out   Printable object.
	 * @param  w     Column width.
	 * @param  j     Column justification.
	 * @param  g     Gap before column.
	 * @param  last  True if this is the last column.
	 * @param  t     Cell text.
	 */
	private static void printCell
		(Printable out,
		 int w,
		 int j,
		 int g,
		 boolean last,
		 String t)
		{
		int before = 0;
		int after = 0;
		if (t.length() > w)
			{
			t = t.substring (0, w);
			}
		else if (j == LEFT)
			{
			after = w - t.length();
			}
		else if (j == RIGHT)
			{
			before = w - t.length();
			}
		else
			{
			before = (w - t.length())/2;
			after = w - before - t.length();
			}
		printSpaces (out, g + before);
		out.print (t);
		if (! last) printSpaces (out, after);
		}

	/**
	 * Print the given number of spaces.
	 *
	 * @param  out  Appendable object.
	 * @param  n    Number of spaces.
	 */
	private static void printSpaces
		(Printable out,
		 int n)
		{
		for (int i = 0; i < n; ++ i)
			out.print (' ');
		}

	}
