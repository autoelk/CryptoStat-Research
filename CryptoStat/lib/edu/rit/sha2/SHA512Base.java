//******************************************************************************
//
// File:    SHA512Base.java
// Package: edu.rit.sha2
// Unit:    Class edu.rit.sha2.SHA512Base
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

import edu.rit.crst.Function;
import edu.rit.util.BigInt;

/**
 * Class SHA512Base is the abstract base class for the SHA-2 hash functions that
 * use a 64-bit word. It implements a 104-byte (832-bit) message. The
 * cryptographic function's input <I>A</I> is the first portion of the message
 * (512 bits); input <I>B</I> is the second portion of the message (320 bits);
 * output <I>C</I> is the digest (size determined by a subclass).
 * <P>
 * Class SHA512Base computes the digest of one 1024-bit block consisting of the
 * message (832 bits) plus the padding (192 bits). The padding consists of one 1
 * bit, sixty-three 0 bits, and the message length (128 bits).
 *
 * @author  Alan Kaminsky
 * @version 25-Aug-2017
 */
public abstract class SHA512Base
	extends Function
	{

// Hidden data members.

	/**
	 * Initialization vector word 0.
	 */
	protected final long IV_0;

	/**
	 * Initialization vector word 1.
	 */
	protected final long IV_1;

	/**
	 * Initialization vector word 2.
	 */
	protected final long IV_2;

	/**
	 * Initialization vector word 3.
	 */
	protected final long IV_3;

	/**
	 * Initialization vector word 4.
	 */
	protected final long IV_4;

	/**
	 * Initialization vector word 5.
	 */
	protected final long IV_5;

	/**
	 * Initialization vector word 6.
	 */
	protected final long IV_6;

	/**
	 * Initialization vector word 7.
	 */
	protected final long IV_7;

	// Round constants.
	private static final long[] K = new long[]
		{
		0x428a2f98d728ae22L, 0x7137449123ef65cdL,
		0xb5c0fbcfec4d3b2fL, 0xe9b5dba58189dbbcL,
		0x3956c25bf348b538L, 0x59f111f1b605d019L,
		0x923f82a4af194f9bL, 0xab1c5ed5da6d8118L,
		0xd807aa98a3030242L, 0x12835b0145706fbeL,
		0x243185be4ee4b28cL, 0x550c7dc3d5ffb4e2L,
		0x72be5d74f27b896fL, 0x80deb1fe3b1696b1L,
		0x9bdc06a725c71235L, 0xc19bf174cf692694L,
		0xe49b69c19ef14ad2L, 0xefbe4786384f25e3L,
		0x0fc19dc68b8cd5b5L, 0x240ca1cc77ac9c65L,
		0x2de92c6f592b0275L, 0x4a7484aa6ea6e483L,
		0x5cb0a9dcbd41fbd4L, 0x76f988da831153b5L,
		0x983e5152ee66dfabL, 0xa831c66d2db43210L,
		0xb00327c898fb213fL, 0xbf597fc7beef0ee4L,
		0xc6e00bf33da88fc2L, 0xd5a79147930aa725L,
		0x06ca6351e003826fL, 0x142929670a0e6e70L,
		0x27b70a8546d22ffcL, 0x2e1b21385c26c926L,
		0x4d2c6dfc5ac42aedL, 0x53380d139d95b3dfL,
		0x650a73548baf63deL, 0x766a0abb3c77b2a8L,
		0x81c2c92e47edaee6L, 0x92722c851482353bL,
		0xa2bfe8a14cf10364L, 0xa81a664bbc423001L,
		0xc24b8b70d0f89791L, 0xc76c51a30654be30L,
		0xd192e819d6ef5218L, 0xd69906245565a910L,
		0xf40e35855771202aL, 0x106aa07032bbd1b8L,
		0x19a4c116b8d2d0c8L, 0x1e376c085141ab53L,
		0x2748774cdf8eeb99L, 0x34b0bcb5e19b48a8L,
		0x391c0cb3c5c95a63L, 0x4ed8aa4ae3418acbL,
		0x5b9cca4f7763e373L, 0x682e6ff3d6b2b8a3L,
		0x748f82ee5defb2fcL, 0x78a5636f43172f60L,
		0x84c87814a1f0ab72L, 0x8cc702081a6439ecL,
		0x90befffa23631e28L, 0xa4506cebde82bde9L,
		0xbef9a3f7b2c67915L, 0xc67178f2e372532bL,
		0xca273eceea26619cL, 0xd186b8c721c0c207L,
		0xeada7dd6cde0eb1eL, 0xf57d4f7fee6ed178L,
		0x06f067aa72176fbaL, 0x0a637dc5a2c898a6L,
		0x113f9804bef90daeL, 0x1b710b35131c471bL,
		0x28db77f523047d84L, 0x32caab7b40c72493L,
		0x3c9ebe0a15c9bebcL, 0x431d67c49c100d4cL,
		0x4cc5d4becb3e42b6L, 0x597f299cfc657e2aL,
		0x5fcb6fab3ad6faecL, 0x6c44198c4a475817L,
		};

	// Message schedule.
	private long[] W = new long [16];

// Exported constructors.

	/**
	 * Construct a new SHA-512 base cryptographic function.
	 *
	 * @param  IV_0  Initialization vector word 0.
	 * @param  IV_1  Initialization vector word 1.
	 * @param  IV_2  Initialization vector word 2.
	 * @param  IV_3  Initialization vector word 3.
	 * @param  IV_4  Initialization vector word 4.
	 * @param  IV_5  Initialization vector word 5.
	 * @param  IV_6  Initialization vector word 6.
	 * @param  IV_7  Initialization vector word 7.
	 */
	public SHA512Base
		(long IV_0,
		 long IV_1,
		 long IV_2,
		 long IV_3,
		 long IV_4,
		 long IV_5,
		 long IV_6,
		 long IV_7)
		{
		super();
		this.IV_0 = IV_0;
		this.IV_1 = IV_1;
		this.IV_2 = IV_2;
		this.IV_3 = IV_3;
		this.IV_4 = IV_4;
		this.IV_5 = IV_5;
		this.IV_6 = IV_6;
		this.IV_7 = IV_7;
		}

// Exported operations.

	/**
	 * Get a description of input <I>A</I> for this cryptographic function.
	 *
	 * @return  Input <I>A</I> description.
	 */
	public String A_description()
		{
		return "message bits 0-511";
		}

	/**
	 * Get the bit size of input <I>A</I> for this cryptographic function.
	 *
	 * @return  Input <I>A</I> bit size.
	 */
	public int A_bitSize()
		{
		return 512;
		}

	/**
	 * Get a description of input <I>B</I> for this cryptographic function.
	 *
	 * @return  Input <I>B</I> description.
	 */
	public String B_description()
		{
		return "message bits 512-831";
		}

	/**
	 * Get the bit size of input <I>B</I> for this cryptographic function.
	 *
	 * @return  Input <I>B</I> bit size.
	 */
	public int B_bitSize()
		{
		return 320;
		}

	/**
	 * Get a description of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> description.
	 */
	public String C_description()
		{
		return "digest";
		}

	/**
	 * Get the number of rounds for this cryptographic function.
	 *
	 * @return  Number of rounds.
	 */
	public int rounds()
		{
		return 80;
		}

	/**
	 * Evaluate this cryptographic function. The function is evaluated on inputs
	 * <I>A</I> and <I>B</I>, and the output of each round is stored in the
	 * <I>C</I> array.
	 * <P>
	 * <I>Note:</I> The <TT>evaluate()</TT> method is performed on the CPU in a
	 * single thread. It is intended as a cross-check for the GPU kernel.
	 *
	 * @param  A  Input <I>A</I>.
	 * @param  B  Input <I>B</I>.
	 * @param  C  Array of outputs <I>C</I>, indexed by round.
	 */
	public void evaluate
		(BigInt A,
		 BigInt B,
		 BigInt[] C)
		{
		// Store message block in message schedule W.
		A.unpackBigEndian (W, 0);
		B.unpackBigEndian (W, 8);
		W[13] = 0x8000000000000000L;
		W[14] = 0L;
		W[15] = 832L;

		// Initialize working variables.
		long a = IV_0;
		long b = IV_1;
		long c = IV_2;
		long d = IV_3;
		long e = IV_4;
		long f = IV_5;
		long g = IV_6;
		long h = IV_7;

		int s;
		long tmp1;
		long tmp2;

		// Do 80 rounds.
		for (int t = 0; t <= 79; ++ t)
			{
			s = t & 15;

			// Compute round function.
			if (t >= 16)
				W[s] = sigma_1 (W[(s+14)&15]) + W[(s+9)&15] +
					sigma_0 (W[(s+1)&15]) + W[s];
			tmp1 = h + Sigma_1(e) + Ch(e,f,g) + K[t] + W[s];
			tmp2 = Sigma_0(a) + Maj(a,b,c);
			h = g;
			g = f;
			f = e;
			e = d + tmp1;
			d = c;
			c = b;
			b = a;
			a = tmp1 + tmp2;

			// Record digest for this round.
			recordDigest (C[t], a, b, c, d, e, f, g, h);
			}
		}

// Hidden operations.

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
	protected abstract void recordDigest
		(BigInt dig,
		 long a,
		 long b,
		 long c,
		 long d,
		 long e,
		 long f,
		 long g,
		 long h);

	/**
	 * The little functions.
	 */
	private static long Ch (long x, long y, long z)
		{
		return (x & y) ^ (~x & z);
		}

	private static long Maj (long x, long y, long z)
		{
		return (x & y) ^ (x & z) ^ (y & z);
		}

	private static long Sigma_0 (long x)
		{
		return Long.rotateRight (x, 28) ^ Long.rotateRight (x, 34) ^
			Long.rotateRight (x, 39);
		}

	private static long Sigma_1 (long x)
		{
		return Long.rotateRight (x, 14) ^ Long.rotateRight (x, 18) ^
			Long.rotateRight (x, 41);
		}

	private static long sigma_0 (long x)
		{
		return Long.rotateRight (x, 1) ^ Long.rotateRight (x, 8) ^
			(x >>> 7);
		}

	private static long sigma_1 (long x)
		{
		return Long.rotateRight (x, 19) ^ Long.rotateRight (x, 61) ^
			(x >>> 6);
		}

	}
