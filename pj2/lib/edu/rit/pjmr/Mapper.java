//******************************************************************************
//
// File:    Mapper.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.Mapper
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

package edu.rit.pjmr;

import edu.rit.pj2.Vbl;

/**
 * Class Mapper is the abstract base class for a mapper in the Parallel Java Map
 * Reduce Framework. See class {@linkplain PjmrJob} for further information
 * about configuring mappers as part of a PJMR job.
 * <P>
 * A {@linkplain MapperTask} calls a mapper's methods as follows:
 * <OL TYPE=1>
 * <P><LI>
 * The mapper task calls the mapper's {@link #start(String[],Combiner) start()}
 * method. The arguments are the mapper's argument strings if any, and the
 * {@linkplain Combiner} that will accumulate the mapper's results. The
 * <TT>start()</TT> method may initialize the mapper based on the argument
 * strings. The <TT>start()</TT> method may do preprocessing operations on the
 * combiner.
 * <P><LI>
 * The mapper task repeatedly calls the mapper's {@link
 * #map(Object,Object,Combiner) map()} method. The arguments are a data record's
 * ID and contents, as well as the {@linkplain Combiner}. The <TT>map()</TT>
 * method extracts an output key and an output value from the data record and
 * adds the (key, value) pair to the combiner. Alternatively, the <TT>map()</TT>
 * method extracts an output key from the data record and adds a (key, null)
 * pair to the combiner; this signifies that the output key exists but has no
 * associated value. Alternatively, the <TT>map()</TT> method decides there is
 * nothing of interest in the data record, and adds nothing to the combiner. The
 * <TT>map()</TT> method may add zero, one, or more than one (key, value) pair
 * to the combiner. The <TT>map()</TT> method may do other operations on the
 * combiner as well.
 * <P><LI>
 * When there are no more data records, the mapper task calls the mapper's
 * {@link #finish(Combiner) finish()} method, passing in the {@linkplain
 * Combiner}. The <TT>finish()</TT> method may do postprocessing operations on
 * the combiner.
 * </OL>
 * <P>
 * If a mapper task has more than one mapper, the mappers are run in parallel in
 * separate threads. There is one global combiner, and each mapper works with
 * its own thread-local copy of the combiner. (A mapper cannot see the other
 * mappers' thread-local combiners.) When the mappers have finished, the
 * thread-local combiners are automatically reduced into the global combiner.
 *
 * @param  <I>  Data type for data record ID.
 * @param  <C>  Data type for data record contents.
 * @param  <K>  Output key data type.
 * @param  <V>  Output value data type; must implement interface {@linkplain
 *              edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 16-Jun-2016
 */
public abstract class Mapper<I,C,K,V extends Vbl>
	{

// Hidden data members.

	Customizer<K,V> customizer;

// Exported constructors.

	/**
	 * Construct a new mapper.
	 */
	public Mapper()
		{
		}

// Exported operations.

	/**
	 * Start this mapper.
	 * <P>
	 * The base class <TT>start()</TT> method does nothing. A subclass may
	 * override the <TT>start()</TT> method to do something.
	 *
	 * @param  args      Array of zero or more argument strings.
	 * @param  combiner  Thread-local combiner.
	 */
	public void start
		(String[] args,
		 Combiner<K,V> combiner)
		{
		}

	/**
	 * Map the given record ID and record contents to a (key, value) pair. The
	 * key must be non-null; the value may be null. The <TT>map()</TT> method
	 * adds the (key, value) pair to the given combiner.
	 * <P>
	 * The <TT>map()</TT> method must be overridden in a subclass.
	 *
	 * @param  id        Data record ID.
	 * @param  contents  Data record contents.
	 * @param  combiner  Thread-local combiner.
	 */
	public abstract void map
		(I id,
		 C contents,
		 Combiner<K,V> combiner);

	/**
	 * Finish this mapper.
	 * <P>
	 * The base class <TT>finish()</TT> method does nothing. A subclass may
	 * override the <TT>finish()</TT> method to do something.
	 *
	 * @param  combiner  Thread-local combiner.
	 */
	public void finish
		(Combiner<K,V> combiner)
		{
		}

	/**
	 * Returns the {@linkplain Customizer Customizer} for this mapper's mapper
	 * task.
	 *
	 * @return  Customizer.
	 */
	public Customizer<K,V> customizer()
		{
		return customizer;
		}

	}
