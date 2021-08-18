//******************************************************************************
//
// File:    OddPrimeList.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.OddPrimeList
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

import edu.rit.util.AList;
import edu.rit.util.BitSet;
import java.math.BigInteger;

/**
 * Class OddPrimeList encapsulates a list of odd prime numbers. The primes are
 * represented as {@linkplain java.math.BigInteger BigInteger}s.
 *
 * @author  Alan Kaminsky
 * @version 02-Jul-2018
 */
public class OddPrimeList
	{

// Hidden data members.

	private int N;
	private int B;
	private BitSet sieve;
	private long p;

	private AList<BigInteger> primes = new AList<BigInteger>();

// Exported constructors.

	/**
	 * Construct a new list of odd primes. The list contains the <I>N</I>
	 * smallest odd primes.
	 *
	 * @param  N  Number of primes.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>N</I> &lt; 1.
	 */
	public OddPrimeList
		(int N)
		{
		// Verify preconditions.
		if (N < 1)
			throw new IllegalArgumentException (String.format
				("OddPrimeList(): N = %d illegal", N));
		this.N = N;

		// Determine upper bound encompassing the N smallest odd primes.
		B = 32;
		while (B/Math.log(B) < N) B *= 2;

		// Initialize sieve with all odd numbers >= 3.
		sieve = new BitSet (B);
		sieve.add (3, B);
		for (int i = 4; i < B; i += 2)
			sieve.remove (i);
		}

// Exported operations.

	/**
	 * Get the number of primes in this odd prime list.
	 *
	 * @return  Number of primes.
	 */
	public int size()
		{
		return N;
		}

	/**
	 * Get the prime at the given index in this odd prime list.
	 *
	 * @param  i  Index.
	 *
	 * @return  Prime at index <TT>i</TT>, 0 &le; <TT>i</TT> &le;
	 *          <I>N</I>&minus;1.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> &lt; is out of bounds.
	 */
	public BigInteger get
		(int i)
		{
		// Verify preconditions.
		if (0 > i || i >= N)
			throw new IndexOutOfBoundsException (String.format
				("OddPrimeList.get(): i = %d out of bounds", i));

		// Find primes and store in list out to index i.
		while (primes.size() <= i)
			{
			p = sieve.nextElement ((int)(p + 1));
			if (p == -1)
				throw new IllegalStateException ("Shouldn't happen");
			primes.addLast (BigInteger.valueOf (p));
			long m = p*p;
			long twop = 2*p;
			while (m < B)
				{
				sieve.remove ((int)m);
				m += twop;
				}
			}

		// Return selected prime.
		return primes.get (i);
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		{
//		// Parse command line arguments.
//		if (args.length != 1) usage();
//		int N = Integer.parseInt (args[0]);
//
//		// Set up list of odd primes.
//		OddPrimeList primes = new OddPrimeList (N);
//
//		// Print primes.
//		for (int i = 0; i < N; ++ i)
//			{
//			BigInteger p = primes.get (i);
//			System.out.printf ("%s", primes.get(i));
//			if (! p.isProbablePrime (64))
//				System.out.printf ("***");
//			System.out.printf (" ");
//			}
//		System.out.println();
//		}
//
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.pj2.example.OddPrimeList <N>");
//		System.err.println ("<N> = Number of odd primes");
//		System.exit (1);
//		}

	}
