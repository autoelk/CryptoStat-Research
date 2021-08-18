//******************************************************************************
//
// File:    CombiningGenerator.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.CombiningGenerator
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
import edu.rit.util.Instance;

/**
 * Class CombiningGenerator is an abstract base class for an object that
 * generates input values for a cryptographic {@linkplain Function Function}.
 * <P>
 * An instance of class CombiningGenerator is layered on top of two or more
 * other input generators. The latter are specified as arguments to the
 * CombiningGenerator constructor, in the form of constructor expressions. The
 * CombiningGenerator constructor uses the constructor expressions to create the
 * underlying input generators. (For further information about constructor
 * expressions, see class {@linkplain edu.rit.util.Instance
 * edu.rit.util.Instance}.)
 * <P>
 * Class CombiningGenerator invokes each underlying input generator to generate
 * a series of cryptographic function input values. Class CombiningGenerator
 * then calls the {@link #combine(GpuBigIntArray[],int) combine()} method to
 * combine the input value series together. The combination is performed in a
 * subclass.
 *
 * @author  Alan Kaminsky
 * @version 20-Sep-2017
 */
public abstract class CombiningGenerator
	extends Generator
	{

// Hidden data members.

	private String[] ctors;
	private Generator[] gens;

// Exported constructors.

	/**
	 * Construct a new combining input generator.
	 *
	 * @param  ctors  Two or more constructor expressions for the underlying
	 *                input generators.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>ctors</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if fewer than two constructor
	 *     expressions are specified.
	 */
	public CombiningGenerator
		(String... ctors)
		{
		super();
		if (ctors == null)
			throw new NullPointerException
				("CombiningGenerator(): ctors is null");
		int N = ctors.length;
		if (N < 2)
			throw new IllegalArgumentException
				("CombiningGenerator(): Two or more constructor expressions required");
		this.ctors = (String[]) ctors.clone();
		this.gens = new Generator [N];
		try
			{
			for (int i = 0; i < N; ++ i)
				this.gens[i] = (Generator) Instance.newInstance (ctors[i]);
			}
		catch (Exception exc)
			{
			throw new IllegalArgumentException (exc);
			}
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
		for (Generator gen : gens)
			gen.setFunction (func, gen_A);
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
		boolean first = true;
		b.append (getClass().getName());
		for (String ctor : ctors)
			{
			b.append (first ? '(' : ',');
			first = false;
			b.append (ctor);
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
		StringBuilder b = new StringBuilder();
		boolean first = true;
		for (Generator gen : gens)
			{
			b.append (first ? "" : " and ");
			first = false;
			b.append ('(');
			b.append (gen.description());
			b.append (')');
			}
		return b.toString();
		}

	/**
	 * Generate a series of values for the designated input of the cryptographic
	 * function.
	 *
	 * @return  Array of input values.
	 */
	public GpuBigIntArray generate()
		{
		int N = gens.length;
		int len = 0;
		GpuBigIntArray[] V = new GpuBigIntArray [N];
		for (int k = 0; k < N; ++ k)
			{
			V[k] = gens[k].generate();
			len += V[k].item.length;
			}
		return combine (V, len);
		}

// Hidden operations.

	/**
	 * Combine the given individual input value sequences together.
	 *
	 * @param  V    Array of {@linkplain GpuBigIntArray} objects. Each element
	 *              of the <TT>V</TT> array contains an input value series
	 *              generated by one of the underlying input generators.
	 * @param  len  Total number of input values in all the input value series.
	 *
	 * @return  A {@linkplain GpuBigIntArray} object containing the combined
	 *          input value series.
	 */
	protected abstract GpuBigIntArray combine
		(GpuBigIntArray[] V,
		 int len);

	}
