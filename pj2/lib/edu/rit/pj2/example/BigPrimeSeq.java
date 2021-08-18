//******************************************************************************
//
// File:    BigPrimeSeq.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.BigPrimeSeq
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

package edu.rit.pj2.example;

import edu.rit.pj2.Task;
import edu.rit.util.Packing;
import java.math.BigInteger;

/**
 * Class BigPrimeSeq is a sequential main program that finds a big prime number.
 * <P>
 * Usage: <TT>java pj2 BigPrimeSeq <I>B</I> <I>N</I> <I>Q</I></TT>
 * <P>
 * The program searches for a Proth prime. A Proth number is of the form
 * <I>k</I>&sdot;2<SUP><I>B</I></SUP>&nbsp;+&nbsp;1, where <I>k</I> is an odd
 * number less than 2<SUP><I>B</I></SUP>. A Proth prime is a Proth number that
 * is prime. Proth's Theorem states that if <I>p</I> is a Proth number, and
 * there exists a number <I>a</I> such that
 * <I>a</I><SUP>(<I>p</I>&minus;1)/2</SUP> &equiv; &minus;1 (mod&nbsp;<I>p</I>),
 * then <I>p</I> is prime.
 * <P>
 * The program does the following for each Proth number <I>p</I> =
 * <I>k</I>&sdot;2<SUP><I>B</I></SUP>&nbsp;+&nbsp;1, <I>k</I> = 1, 3, 5, ...:
 * <OL TYPE=1>
 * <P><LI>
 * Do a trial division test on <I>p</I>: Check whether <I>p</I> has a factor
 * among the smallest <I>N</I> primes. If so, try the next <I>k</I>. If not, go
 * to Step 2.
 * <P><LI>
 * Do a Proth test on <I>p</I>: Check whether a number <I>a</I> among the
 * smallest <I>Q</I> primes satisfies the formula in Proth's Theorem. If not,
 * try the next <I>k</I>. If so, report the Proth prime <I>p</I> and stop.
 * </OL>
 * <P>
 * To indicate progress during the run, the program prints each Proth number to
 * which the program applies the Proth test. At the end of the run, the program
 * prints the number of Proth numbers examined and the number of Proth tests
 * performed.
 *
 * @author  Alan Kaminsky
 * @version 20-Jun-2018
 */
public class BigPrimeSeq
	extends Task
	{
	final BigInteger ZERO = new BigInteger ("0");
	final BigInteger ONE  = new BigInteger ("1");
	final BigInteger TWO  = new BigInteger ("2");

	int B;
	int N;
	int Q;
	OddPrimeList primes;

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 3) usage();
		B = Integer.parseInt (args[0]);
		N = Integer.parseInt (args[1]);
		Q = Integer.parseInt (args[2]);
		if (B < 1) error ("<B> must be >= 1");
		if (Q > N) error ("<Q> must be <= <N>");

		// Set up list of odd primes.
		primes = new OddPrimeList (N);

		// Test Proth numbers (k*2^B + 1).
		int count = 0;
		int potentialCount = 0;
		int k = 1;
		for (;;)
			{
			++ count;
			BigInteger p = prothNumber (k, B);
			if (trialDivisionTest (p))
				{
				++ potentialCount;
				if (prothTest (p))
					{
					System.out.printf ("%d*2^%d + 1 is prime%n", k, B);
					System.out.println (p);
					break;
					}
				else
					{
					System.out.printf ("%d*2^%d + 1%n", k, B);
					System.out.flush();
					}
				}
			k += 2;
			}

		// Print results.
		System.out.printf ("%d Proth numbers%n", count);
		System.out.printf ("%d potential primes%n", potentialCount);
		}

	/**
	 * Returns true if p has no factors in the odd prime list.
	 */
	private boolean trialDivisionTest
		(BigInteger p)
		{
		for (int i = 0; i < N; ++ i)
			if (p.mod (primes.get(i)) .compareTo (ZERO) == 0)
				return false;
		return true;
		}

	/**
	 * Returns true if Proth number p can be shown to be prime.
	 */
	private boolean prothTest
		(BigInteger p)
		{
		BigInteger exp = p.subtract (ONE) .divide (TWO);
		for (int i = 0; i < Q; ++ i)
			if (primes.get(i) .modPow (exp, p) .add (ONE) .compareTo (p) == 0)
				return true;
		return false;
		}

	/**
	 * Print an error message and terminate.
	 */
	private static void error
		(String msg)
		{
		System.err.printf ("BigPrimeSeq: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 BigPrimeSeq <B> <N> <Q>");
		System.err.println ("<B> = Bit size; >= 1");
		System.err.println ("<N> = Number of primes in trial division test");
		System.err.println ("<Q> = Number of bases in Proth test; <= N");
		terminate (1);
		}

	/**
	 * Returns the Proth number k*2^b + 1.
	 */
	private static BigInteger prothNumber
		(int k,
		 int b)
		{
		return BigInteger.valueOf (k) .shiftLeft (b) .setBit (0);
		}

	/**
	 * This task requires one core.
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	}
