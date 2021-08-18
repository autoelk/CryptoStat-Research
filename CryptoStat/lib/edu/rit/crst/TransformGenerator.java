//******************************************************************************
//
// File:    TransformGenerator.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.TransformGenerator
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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
import edu.rit.util.GpuBigIntArray;
import edu.rit.util.Instance;

/**
 * Class TransformGenerator is the abstract base class for an object that
 * generates input values for a cryptographic {@linkplain Function Function}.
 * <P>
 * An instance of class TransformGenerator is layered on top of another input
 * generator. The latter is specified as an argument to the TransformGenerator
 * constructor, in the form of a constructor expression. The TransformGenerator
 * constructor uses the constructor expression to create the underlying input
 * generator. (For further information about constructor expressions, see class
 * {@linkplain edu.rit.util.Instance edu.rit.util.Instance}.)
 * <P>
 * Class TransformGenerator invokes the underlying input generator to generate a
 * series of cryptographic function input values. Class TransformGenerator then
 * calls the {@link #transform(BigInt) transform()} method on each input value,
 * and the result replaces that input value. The transformation is defined in a
 * subclass.
 *
 * @author  Alan Kaminsky
 * @version 15-Nov-2016
 */
public abstract class TransformGenerator
	extends Generator
	{

// Hidden data members.

	String ctor;
	Generator gen;

// Exported constructors.

	/**
	 * Construct a new transformed input generator.
	 *
	 * @param  ctor  Constructor expression for the underlying input generator.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>ctor</TT> is null.
	 */
	public TransformGenerator
		(String ctor)
		{
		super();
		if (ctor == null)
			throw new NullPointerException
				("TransformGenerator(): ctor is null");
		this.ctor = ctor;
		try
			{
			this.gen = (Generator) Instance.newInstance (ctor);
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
		gen.setFunction (func, gen_A);
		}

	/**
	 * Generate a series of values for the designated input of the cryptographic
	 * function.
	 *
	 * @return  Array of input values.
	 */
	public GpuBigIntArray generate()
		{
		GpuBigIntArray V = gen.generate();
		for (int i = 0; i < V.item.length; ++ i)
			transform (V.item[i]);
		return V;
		}

// Hidden operations.

	/**
	 * Transform the given input value.
	 * <P>
	 * The <TT>transform()</TT> method must be defined in a subclass.
	 *
	 * @param  value  Input value.
	 */
	protected abstract void transform
		(BigInt value);

	}
