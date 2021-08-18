//******************************************************************************
//
// File:    Permute.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Permute
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

import edu.rit.util.BigInt;
import edu.rit.util.Random;
import edu.rit.util.RandomSubset;

/**
 * Class Permute provides an object that generates input values for a
 * cryptographic {@linkplain Function Function}.
 * <P>
 * An instance of class Permute is layered on top of another input generator.
 * The latter is specified as an argument to the Permute constructor, in the
 * form of a constructor expression. The Permute constructor uses the
 * constructor expression to create the underlying input generator. (For further
 * information about constructor expressions, see class {@linkplain
 * edu.rit.util.Instance edu.rit.util.Instance}.)
 * <P>
 * Class Permute invokes the underlying input generator to generate a series of
 * cryptographic function input values. Class Permute then applies a bit
 * permutation to each input value. The bit permutation is specified by the
 * Permute constructor's arguments.
 *
 * @author  Alan Kaminsky
 * @version 25-Apr-2017
 */
public class Permute
	extends TransformGenerator
	{

// Hidden data members.

	private int[] perm;
	private long seed;
	private Random prng;

// Exported constructors.

	/**
	 * Construct a new permute input generator that swaps the given bit
	 * positions. The bits at position 0 and position <TT>perm[0]</TT> are
	 * swapped, the bits at position 1 and position <TT>perm[1]</TT> are
	 * swapped, and so on.
	 *
	 * @param  ctor  Constructor expression for the underlying input generator.
	 * @param  perm  Zero or more bit positions to swap.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>ctor</TT> is null. Thrown if
	 *     <TT>perm</TT> is null.
	 */
	public Permute
		(String ctor,
		 int... perm)
		{
		super (ctor);
		if (perm == null)
			throw new NullPointerException ("Permute(): perm is null");
		this.perm = (int[]) perm.clone();
		}

	/**
	 * Construct a new permute input generator that swaps randomly chosen bit
	 * positions. Each bit position from 0 through <TT>n</TT>&minus;1 is swapped
	 * with a randomly chosen bit position.
	 *
	 * @param  ctor  Constructor expression for the underlying input generator.
	 * @param  n     Number of bit positions to swap.
	 * @param  seed  Pseudorandom number generator seed.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>ctor</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>n</TT> &lt; 0.
	 */
	public Permute
		(String ctor,
		 int n,
		 long seed)
		{
		super (ctor);
		if (n < 0)
			throw new IllegalArgumentException (String.format
				("Permute(): n = %d illegal", n));
		this.perm = new int [n];
		this.seed = seed;
		this.prng = new Random (seed);
		}

// Exported operations.

	/**
	 * Set the cryptographic function for this input generator.
	 *
	 * @param  func   Cryptographic function object.
	 * @param  gen_A  True to generate A input values, false to generate B input
	 *                values.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>func</TT> is null.
	 */
	public void setFunction
		(Function func,
		 boolean gen_A)
		{
		super.setFunction (func, gen_A);
		if (prng == null)
			{ }
		else if (gen_A)
			setPerm (func.A_bitSize());
		else
			setPerm (func.B_bitSize());
		}

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
		StringBuilder b = new StringBuilder();
		b.append ("Permute(");
		b.append (ctor);
		if (prng == null)
			{
			if (perm.length > 0)
				{
				b.append (',');
				b.append (permToString());
				}
			}
		else
			{
			b.append (',');
			b.append (perm.length);
			b.append (',');
			b.append (seed);
			b.append ('L');
			}
		b.append (')');
		return b.toString();
		}

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return gen.description() + ", permuted (" + permToString() + ")";
		}

// Hidden operations.

	/**
	 * Transform the given input value. The result replaces that input value.
	 *
	 * @param  value  Input value.
	 */
	protected void transform
		(BigInt value)
		{
		for (int i = 0; i < perm.length; ++ i)
			{
			int a = value.getBit (i);
			int b = value.getBit (perm[i]);
			value.putBit (i, b);
			value.putBit (perm[i], a);
			}
		}

	/**
	 * Set the bit permutation at random.
	 *
	 * @param  bitSize  Cryptographic function's input bit size.
	 */
	private void setPerm
		(int bitSize)
		{
		RandomSubset rs = new RandomSubset (prng, bitSize, true);
		for (int i = 0; i < perm.length; ++ i)
			perm[i] = rs.next();
		}

	/**
	 * Returns a string version of the bit permutation.
	 */
	private String permToString()
		{
		StringBuilder b = new StringBuilder();
		for (int i = 0; i < perm.length; ++ i)
			{
			if (i > 0) b.append (',');
			b.append (perm[i]);
			}
		return b.toString();
		}

	}
