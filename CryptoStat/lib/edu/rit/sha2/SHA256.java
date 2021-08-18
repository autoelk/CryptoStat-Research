//******************************************************************************
//
// File:    SHA256.java
// Package: edu.rit.sha2
// Unit:    Class edu.rit.sha2.SHA256
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

package edu.rit.sha2;

import edu.rit.util.BigInt;

/**
 * Class SHA256 implements the SHA-256 hash function with a 52-byte (416-bit)
 * message. The SHA256 cryptographic function's input <I>A</I> is the first
 * portion of the message (256 bits); input <I>B</I> is the second portion of
 * the message (160 bits); output <I>C</I> is the digest (256 bits).
 * <P>
 * Class SHA256 computes the digest of one 512-bit block consisting of the
 * message (416 bits) plus the padding (96 bits). The padding consists of one 1
 * bit, thirty-one 0 bits, and the message length (64 bits).
 *
 * @author  Alan Kaminsky
 * @version 25-Aug-2017
 */
public class SHA256
	extends SHA256Base
	{

// Exported constructors.

	/**
	 * Construct a new SHA-256 cryptographic function.
	 */
	public SHA256()
		{
		super
			(0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
			 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19);
		}

// Exported operations.

	/**
	 * Get a constructor expression for this cryptographic function. The
	 * constructor expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this cryptographic function.
	 *
	 * @return  Constructor expression.
	 */
	public String constructor()
		{
		return "edu.rit.sha2.SHA256()";
		}

	/**
	 * Get a description of this cryptographic function.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "SHA-256 hash function";
		}

	/**
	 * Get the bit size of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> bit size.
	 */
	public int C_bitSize()
		{
		return 256;
		}

// Hidden operations.

	/**
	 * Get the GPU kernel module name for this computation object.
	 *
	 * @return  Module name.
	 */
	protected String moduleName()
		{
		return "edu/rit/sha2/SHA256.ptx";
		}

	/**
	 * Record the current round digest.
	 *
	 * @param  dig  Digest.
	 * @param  a    Working variable a.
	 * @param  b    Working variable b.
	 * @param  c    Working variable c.
	 * @param  d    Working variable d.
	 * @param  e    Working variable e.
	 * @param  f    Working variable f.
	 * @param  g    Working variable g.
	 * @param  h    Working variable h.
	 */
	protected void recordDigest
		(BigInt dig,
		 int a,
		 int b,
		 int c,
		 int d,
		 int e,
		 int f,
		 int g,
		 int h)
		{
		dig.value[7] = a + IV_0;
		dig.value[6] = b + IV_1;
		dig.value[5] = c + IV_2;
		dig.value[4] = d + IV_3;
		dig.value[3] = e + IV_4;
		dig.value[2] = f + IV_5;
		dig.value[1] = g + IV_6;
		dig.value[0] = h + IV_7;
		}

	}
