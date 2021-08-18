//******************************************************************************
//
// File:    Eval.java
// Unit:    Class Eval
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the CryptoStat Library ("CryptoStat").
// CryptoStat is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// CryptoStat is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

import edu.rit.crst.Function;
import edu.rit.crst.TestKernel;
import edu.rit.util.BigInt;
import edu.rit.util.Instance;

/**
 * Class Eval is a program that evaluates a cryptographic function for a single
 * input. It is intended for unit testing the implementation of the
 * cryptographic function.
 * <P>
 * Usage: <TT>java Eval "<I>function</I>" <I>A</I> <I>B</I></TT>
 * <P>
 * The <I>function</I> argument is a constructor expression for the {@linkplain
 * Function Function} object that implements the cryptographic function to be
 * tested. (For further information about constructor expressions, see class
 * {@linkplain edu.rit.util.Instance edu.rit.util.Instance}.)
 * <P>
 * The <I>A</I> argument is the function's A input value (hexadecimal).
 * <P>
 * The <I>B</I> argument is the function's B input value (hexadecimal).
 * <P>
 * The Eval program computes and prints the function's C output (hexadecimal)
 * for each round of the function.
 *
 * @author  Alan Kaminsky
 * @version 21-Aug-2017
 */
public class Eval
	{

	/**
	 * Main program.
	 */
	public static void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 3) usage();
		Function func = (Function) Instance.newInstance (args[0]);
		BigInt A = new BigInt (func.A_bitSize()) .fromString (args[1]);
		BigInt B = new BigInt (func.B_bitSize()) .fromString (args[2]);

		// Set up array of C outputs.
		int R = func.rounds();
		BigInt[] C = new BigInt [R];
		for (int i = 0; i < R; ++ i)
			C[i] = new BigInt (func.C_bitSize());

		// Evaluate function.
		func.evaluate (A, B, C);

		// Print function outputs.
		for (int i = 0; i < R; ++ i)
			System.out.printf ("%d\t%s\n", i, C[i]);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java Eval \"<function>\" <A> <B>");
		System.err.println ("<function> = Cryptographic function constructor expression");
		System.err.println ("<A> = A input value (hexadecimal)");
		System.err.println ("<B> = B input value (hexadecimal)");
		System.exit (1);
		}

	}
