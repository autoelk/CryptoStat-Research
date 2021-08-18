//******************************************************************************
//
// File:    Rand.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Rand
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

package edu.rit.crst;

import edu.rit.util.GpuBigIntArray;
import edu.rit.util.Random;

/**
 * Class Rand provides an object that generates input values for a cryptographic
 * {@linkplain Function Function}.
 * <P>
 * Class Seq generates <I>N</I> cryptographic function input values, where
 * <I>N</I> is a constructor argument. The input values are chosen uniformly at
 * random in the range 0 to 2<SUP><I>B</I></SUP>&minus;1, where <I>B</I> is a
 * constructor argument; the default <I>B</I> is the cryptographic function's
 * input bit size.
 *
 * @author  Alan Kaminsky
 * @version 24-Oct-2017
 */
public class Rand
	extends Generator
	{

// Hidden data members.

	private int N;
	private long seed;
	private Integer B;

// Exported constructors.

	/**
	 * Construct a new random input generator. The input values are chosen
	 * uniformly at random in the range 0 to 2<SUP><I>B</I></SUP>&minus;1, where
	 * <I>B</I> is the cryptographic function's input bit size.
	 *
	 * @param  N     Number of input values to generate.
	 * @param  seed  Pseudorandom number generator seed.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>N</TT> &lt; 1.
	 */
	public Rand
		(int N,
		 long seed)
		{
		super();
		if (N < 1)
			throw new IllegalArgumentException (String.format
				("Rand(): N = %d illegal", N));
		this.N = N;
		this.seed = seed;
		}

	/**
	 * Construct a new random input generator with values in the given range.
	 * The input values are chosen uniformly at random in the range 0 to
	 * 2<SUP><I>B</I></SUP>&minus;1, where <I>B</I> is the smaller of the
	 * argument <TT>B</TT> and the cryptographic function's input bit size.
	 *
	 * @param  N     Number of input values to generate.
	 * @param  seed  Pseudorandom number generator seed.
	 * @param  B     Number of bits to choose at random.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>N</TT> &lt; 1. Thrown if
	 *     <TT>B</TT> &lt; 0.
	 */
	public Rand
		(int N,
		 long seed,
		 int B)
		{
		this (N, seed);
		if (B < 0)
			throw new IllegalArgumentException (String.format
				("Rand(): B = %d illegal", B));
		this.B = B;
		}

// Exported operations.

	/**
	 * Get a constructor expression for this input generator. The constructor
	 * expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this input generator.
	 *
	 * @return  Constructor expression.
	 */
	public String constructor()
		{
		if (B == null)
			return String.format ("Rand(%d,%d)", N, seed);
		else
			return String.format ("Rand(%d,%d,%s)", N, seed, B);
		}

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "Random inputs";
		}

	/**
	 * Generate a series of values for the designated input of the cryptographic
	 * function.
	 *
	 * @return  Array of input values.
	 */
	public GpuBigIntArray generate()
		{
		int bitSize = gen_A ? func.A_bitSize() : func.B_bitSize();
		int randSize = (B == null) ? bitSize : Math.min (bitSize, B);
		Random prng = new Random (seed);
		GpuBigIntArray V = new GpuBigIntArray (bitSize, N);
		for (int i = 0; i < N; ++ i)
			V.item[i] .randomize (prng, randSize);
		return V;
		}

	}
