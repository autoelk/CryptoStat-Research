//******************************************************************************
//
// File:    PrintGraph.java
// Package: edu.rit.util.test
// Unit:    Class edu.rit.util.test.PrintGraph
//
// This Java source file is copyright (C) 2018 by Alan Kaminsky. All rights
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

package edu.rit.util.test;

import edu.rit.util.GraphSpec;
import edu.rit.util.Instance;

/**
 * Class PrintGraph is a main program that prints a graph specified by a
 * {@linkplain edu.rit.util.GraphSpec GraphSpec} object. The program constructs
 * an instance of the graph specification class using the constructor expression
 * on the command line. The program prints the following:
 * <PRE>
 *     C &lt;ctor&gt;
 *     G &lt;V&gt; &lt;E&gt;
 *     E &lt;v1&gt; &lt;v2&gt;
 *     . . .
 * </PRE>
 * where <TT>&lt;ctor&gt;</TT> is the GraphSpec constructor expression,
 * <TT>&lt;V&gt;</TT> is the number of vertices, <TT>&lt;E&gt;</TT> is the
 * number of edges, and <TT>&lt;v1&gt;</TT>, <TT>&lt;v2&gt;</TT> are the
 * vertices connected by one of the edges.
 * <P>
 * Usage: <TT>java edu.rit.util.test.PrintGraph "<I>ctor</I>"</TT>
 * <BR><TT><I>ctor</I></TT> = GraphSpec constructor expression
 *
 * @author  Alan Kaminsky
 * @version 26-Apr-2018
 */
public class PrintGraph
	{

// Prevent construction.

	private PrintGraph()
		{
		}

// Main program.

	/**
	 * Main program.
	 */
	public static void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 1) usage();
		String ctor = args[0];

		// Construct graph spec object.
		GraphSpec gspec = (GraphSpec) Instance.newInstance (ctor);

		// Print graph.
		System.out.printf ("C %s%n", ctor);
		System.out.printf ("G %d %d%n", gspec.V(), gspec.E());
		while (gspec.hasNext())
			{
			GraphSpec.Edge edge = gspec.next();
			System.out.printf ("E %d %d%n", edge.v1, edge.v2);
			}
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java edu.rit.util.test.PrintGraph \"<ctor>\"");
		System.err.println ("<ctor> = GraphSpec constructor expression");
		System.exit (1);
		}

	}
