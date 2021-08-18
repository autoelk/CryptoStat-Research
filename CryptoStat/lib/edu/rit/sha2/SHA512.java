//******************************************************************************
//
// File:    SHA512.java
// Package: edu.rit.sha2
// Unit:    Class edu.rit.sha2.SHA512
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

import edu.rit.gpu.Gpu;
import edu.rit.util.BigInt;
import java.io.IOException;

/**
 * Class SHA512 implements the SHA-512 hash function with a 104-byte (832-bit)
 * message. The SHA512 cryptographic function's input <I>A</I> is the first
 * portion of the message (512 bits); input <I>B</I> is the second portion of
 * the message (320 bits); output <I>C</I> is the digest (512 bits).
 * <P>
 * Class SHA512 computes the digest of one 1024-bit block consisting of the
 * message (832 bits) plus the padding (192 bits). The padding consists of one 1
 * bit, sixty-three 0 bits, and the message length (128 bits).
 *
 * @author  Alan Kaminsky
 * @version 25-Aug-2017
 */
public class SHA512
	extends SHA512Base
	{

// Exported constructors.

	/**
	 * Construct a new SHA-512 cryptographic function.
	 */
	public SHA512()
		{
		super
			(0x6a09e667f3bcc908L, 0xbb67ae8584caa73bL,
			 0x3c6ef372fe94f82bL, 0xa54ff53a5f1d36f1L,
			 0x510e527fade682d1L, 0x9b05688c2b3e6c1fL,
			 0x1f83d9abfb41bd6bL, 0x5be0cd19137e2179L);
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
		return "edu.rit.sha2.SHA512()";
		}

	/**
	 * Get a description of this cryptographic function.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "SHA-512 hash function";
		}

	/**
	 * Get the bit size of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> bit size.
	 */
	public int C_bitSize()
		{
		return 512;
		}

// Hidden operations.

	/**
	 * Get the GPU kernel module name for this computation object.
	 *
	 * @return  Module name.
	 */
	protected String moduleName()
		{
		return "edu/rit/sha2/SHA512.ptx";
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
		 long a,
		 long b,
		 long c,
		 long d,
		 long e,
		 long f,
		 long g,
		 long h)
		{
		long tmp;
		tmp = a + IV_0;
		dig.value[15] = (int)(tmp >> 32);
		dig.value[14] = (int)(tmp);
		tmp = b + IV_1;
		dig.value[13] = (int)(tmp >> 32);
		dig.value[12] = (int)(tmp);
		tmp = c + IV_2;
		dig.value[11] = (int)(tmp >> 32);
		dig.value[10] = (int)(tmp);
		tmp = d + IV_3;
		dig.value[ 9] = (int)(tmp >> 32);
		dig.value[ 8] = (int)(tmp);
		tmp = e + IV_4;
		dig.value[ 7] = (int)(tmp >> 32);
		dig.value[ 6] = (int)(tmp);
		tmp = f + IV_5;
		dig.value[ 5] = (int)(tmp >> 32);
		dig.value[ 4] = (int)(tmp);
		tmp = g + IV_6;
		dig.value[ 3] = (int)(tmp >> 32);
		dig.value[ 2] = (int)(tmp);
		tmp = h + IV_7;
		dig.value[ 1] = (int)(tmp >> 32);
		dig.value[ 0] = (int)(tmp);
		}

	}
