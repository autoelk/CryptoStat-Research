//******************************************************************************
//
// File:    NodeProperties.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.NodeProperties
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
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

package edu.rit.pj2.tracker;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class NodeProperties encapsulates the capabilities a computational node must
 * have in order to run a certain PJ2 {@linkplain edu.rit.pj2.Task Task} as part
 * of a PJ2 {@linkplain edu.rit.pj2.Job Job}. The following capabilities may be
 * specified:
 * <UL>
 * <LI><TT>nodeName</TT> &mdash; The name of the node needed.
 * <LI><TT>cores</TT> &mdash; The number of CPU cores needed.
 * <LI><TT>gpus</TT> &mdash; The number of GPU accelerators needed.
 * <LI><TT>gpuIds</TT> &mdash; The specific GPU IDs needed.
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 27-Jan-2017
 */
public class NodeProperties
	implements Streamable
	{

// Exported constants.

	/**
	 * Indicates that the <TT>nodeName</TT> property is defaulted.
	 */
	public static final String DEFAULT_NODE_NAME = null;

	/**
	 * Indicates that the <TT>cores</TT> property is defaulted.
	 */
	public static final int DEFAULT_CORES = -2;

	/**
	 * Indicates that the <TT>gpus</TT> property is defaulted.
	 */
	public static final int DEFAULT_GPUS = -2;

	/**
	 * Indicates that the task can run on any node of the cluster.
	 */
	public static final String ANY_NODE_NAME = "";

	/**
	 * Indicates that the task requires all the cores on the node.
	 */
	public static final int ALL_CORES = -1;

	/**
	 * Indicates that the task requires all the GPU accelerators on the node.
	 */
	public static final int ALL_GPUS = -1;

// Hidden data members.

	String nodeName = DEFAULT_NODE_NAME;
	int cores = DEFAULT_CORES;
	int gpus = DEFAULT_GPUS;
	int[] gpuIDs = null;

// Exported constructors.

	/**
	 * Construct a new node properties object. All settings are defaulted.
	 */
	public NodeProperties()
		{
		}

	/**
	 * Construct a new node properties object that is a copy of the given node
	 * properties object.
	 *
	 * @param  node  Node properties object to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>node</TT> is null.
	 */
	public NodeProperties
		(NodeProperties node)
		{
		this.nodeName = node.nodeName;
		this.cores = node.cores;
		this.gpus = node.gpus;
		this.gpuIDs =
			(node.gpuIDs == null) ? null : (int[]) node.gpuIDs.clone();
		}

	/**
	 * Construct a new node properties object with capabilities specified by the
	 * given string. The <TT>capabilities</TT> string must be in the format
	 * <TT>"<I>name,cores,gpus</I>"</TT>, where <TT><I>name</I></TT> is the name
	 * of the node, <TT><I>cores</I></TT> is the number of CPU cores in the node
	 * (&ge; 1), and <TT><I>gpus</I></TT> is either an integer giving the number
	 * of GPU accelerators in the node (&ge; 0) or a parenthesized
	 * comma-separated list of one or more integers giving the IDs of the GPU
	 * accelerators in the node.
	 *
	 * @param  capabilities  Capabilities string.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>capabilities</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>capabilities</TT> is illegal.
	 */
	public NodeProperties
		(String capabilities)
		{
		// Split capabilities string into 3 pieces at the commas.
		String[] token = capabilities.split (",", 3);

		// Parse node name.
		if (token.length < 1)
			throw new IllegalArgumentException
				("NodeProperties(): Node name missing");
		nodeName (token[0]);
		if (nodeName.length() == 0)
			throw new IllegalArgumentException
				("NodeProperties(): nodeName = \"\" illegal");

		// Parse CPU cores.
		if (token.length < 2)
			throw new IllegalArgumentException
				("NodeProperties(): Cores missing");
		cores (token[1]);
		if (cores < 1)
			throw new IllegalArgumentException (String.format
				("NodeProperties(): cores = %d illegal", cores));

		// Parse GPU accelerators.
		if (token.length < 3)
			throw new IllegalArgumentException
				("NodeProperties(): GPUs missing");
		gpus (token[2]);
		if (gpus < 0)
			throw new IllegalArgumentException (String.format
				("NodeProperties(): gpus = %d illegal", gpus));
		}

// Exported operations.

	/**
	 * Set the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which the task must run.
	 *
	 * @param  nodeName  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *                   #DEFAULT_NODE_NAME}.
	 *
	 * @return  This node properties object.
	 *
	 * @see  #nodeName()
	 */
	public NodeProperties nodeName
		(String nodeName)
		{
		this.nodeName = nodeName;
		return this;
		}

	/**
	 * Get the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which the task must run. If the
	 * <TT>nodeName</TT> property is defaulted, {@link #ANY_NODE_NAME} is
	 * returned, indicating that the task can run on any node of the cluster.
	 *
	 * @return  Node name, or {@link #ANY_NODE_NAME}.
	 *
	 * @see  #nodeName(String)
	 */
	public String nodeName()
		{
		return nodeName == DEFAULT_NODE_NAME ? ANY_NODE_NAME : nodeName;
		}

	/**
	 * Set the <TT>cores</TT> property from the given string. The <TT>cores</TT>
	 * property specifies the number of CPU cores the task requires.
	 *
	 * @param  s  Number of cores (integer string &ge; 1).
	 *
	 * @return  This node properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> is illegal.
	 *
	 * @see  #cores()
	 */
	public NodeProperties cores
		(String s)
		{
		try
			{
			return cores (Integer.parseInt (s));
			}
		catch (NumberFormatException exc)
			{
			throw new IllegalArgumentException (String.format
				("NodeProperties.cores(): s = \"%s\" illegal", s));
			}
		}

	/**
	 * Set the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores the task requires.
	 *
	 * @param  cores  Number of cores (&ge; 1), {@link #ALL_CORES}, or {@link
	 *                #DEFAULT_CORES}.
	 *
	 * @return  This node properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>cores</TT> is illegal.
	 *
	 * @see  #cores()
	 */
	public NodeProperties cores
		(int cores)
		{
		if (cores < DEFAULT_CORES || cores == 0)
			throw new IllegalArgumentException (String.format
				("NodeProperties.cores(): cores = %d illegal", cores));
		this.cores = cores;
		return this;
		}

	/**
	 * Get the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores the task requires. If the <TT>cores</TT> property
	 * is defaulted, {@link #ALL_CORES} is returned, indicating that the task
	 * requires all the cores on the node.
	 *
	 * @return  Number of cores (&ge; 1), or {@link #ALL_CORES}.
	 *
	 * @see  #cores(int)
	 */
	public int cores()
		{
		return cores == DEFAULT_CORES ? ALL_CORES : cores;
		}

	/**
	 * Set the <TT>gpus</TT> and <TT>gpuIDs</TT> properties from the given
	 * string. If <TT>s</TT> is a single integer &ge; 0, it gives the number of
	 * GPU accelerators the task requires, and any GPU IDs may be used. If
	 * <TT>s</TT> is a parenthesized comma-separated list of one or more
	 * integers, it gives the specific GPU IDs the task requires.
	 *
	 * @param  s  Number of GPUs (integer string &ge; 0), or list of GPU IDs.
	 *
	 * @return  This node properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> is illegal.
	 *
	 * @see  #gpus()
	 */
	public NodeProperties gpus
		(String s)
		{
		try
			{
			if (s.startsWith ("(") && s.endsWith (")"))
				{
				// Parse list of GPU IDs.
				String[] id = s.substring (1, s.length()-1) .split (",");
				gpus = id.length;
				gpuIDs = new int [gpus];
				for (int i = 0; i < gpus; ++ i)
					gpuIDs[i] = Integer.parseInt (id[i]);
				}
			else
				{
				// Parse number of GPUs.
				gpus = Integer.parseInt (s);
				if (gpus < 0)
					throw new NumberFormatException();
				gpuIDs = new int [gpus];
				for (int i = 0; i < gpus; ++ i)
					gpuIDs[i] = i;
				}
			return this;
			}
		catch (NumberFormatException exc)
			{
			throw new IllegalArgumentException (String.format
				("NodeProperties.gpus(): s = \"%s\" illegal", s));
			}
		}

	/**
	 * Set the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators the task requires. Any GPU IDs may be used.
	 *
	 * @param  gpus  Number of GPUs (&ge; 0), {@link #ALL_GPUS}, or {@link
	 *               #DEFAULT_GPUS}.
	 *
	 * @return  This node properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gpus</TT> is illegal.
	 *
	 * @see  #gpus()
	 */
	public NodeProperties gpus
		(int gpus)
		{
		if (gpus < DEFAULT_GPUS)
			throw new IllegalArgumentException (String.format
				("NodeProperties.gpus(): gpus = %d illegal", gpus));
		this.gpus = gpus;
		this.gpuIDs = null;
		return this;
		}

	/**
	 * Get the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators the task requires. If the <TT>gpus</TT>
	 * property is defaulted, 0 is returned, indicating that the task requires
	 * no GPU accelerators.
	 *
	 * @return  Number of GPUs (&ge; 0).
	 *
	 * @see  #gpus(int)
	 */
	public int gpus()
		{
		return gpus == DEFAULT_GPUS ? 0 : gpus;
		}

	/**
	 * Set the <TT>gpuIDs</TT> property. The <TT>gpuIDs</TT> property specifies
	 * the GPU IDs the task requires.
	 *
	 * @param  gpuIDs  Array of GPU IDs the task requires. If null or
	 *                 zero-length, the task requires no GPU accelerators.
	 *
	 * @return  This node properties object.
	 *
	 * @see  #gpuIDs()
	 */
	public NodeProperties gpuIDs
		(int[] gpuIDs)
		{
		if (gpuIDs == null || gpuIDs.length == 0)
			{
			this.gpus = 0;
			this.gpuIDs = null;
			}
		else
			{
			this.gpus = gpuIDs.length;
			this.gpuIDs = (int[]) gpuIDs.clone();
			}
		return this;
		}

	/**
	 * Get the <TT>gpuIDs</TT> property. The <TT>gpuIDs</TT> property specifies
	 * the GPU IDs the task requires. If the task requires no GPU accelerators,
	 * or if the task may use any GPU IDs, null is returned.
	 *
	 * @return  Array of GPU IDs the task requires, or null.
	 *
	 * @see  #gpuIDs(int[])
	 */
	public int[] gpuIDs()
		{
		return gpuIDs;
		}

	/**
	 * Write this object's fields to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeString (nodeName);
		out.writeInt (cores);
		out.writeInt (gpus);
		out.writeIntArray (gpuIDs);
		}

	/**
	 * Read this object's fields from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		nodeName = in.readString();
		cores = in.readInt();
		gpus = in.readInt();
		gpuIDs = in.readIntArray();
		}

	/**
	 * Returns a string version of this node capabilities object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		StringBuilder b = new StringBuilder();
		b.append ("NodeProperties(\"");
		b.append (nodeName());
		b.append ("\",");
		b.append (cores());
		b.append (",");
		int[] id = gpuIDs();
		if (id == null)
			b.append (gpus());
		else
			{
			b.append ("(");
			for (int i = 0; i < id.length; ++ i)
				{
				if (i > 0) b.append (",");
				b.append (id[i]);
				}
			b.append (")");
			}
		b.append (")");
		return b.toString();
		}

	}
