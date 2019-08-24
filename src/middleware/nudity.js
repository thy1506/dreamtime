/*
 * DreamTime | (C) 2019 by Ivan Bravo Bravo <ivan@dreamnet.tech>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License 3.0 as published by
 * the Free Software Foundation.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

import _ from 'lodash'

/**
 * Pages that include this middleware will only be accessible
 * if the user has selected a photo to transform.
 */
export default function({ redirect, $nudify }) {
  if (!$nudify.hasPhoto()) {
    redirect('/')
  }
}
