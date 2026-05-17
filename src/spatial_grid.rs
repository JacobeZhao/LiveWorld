// Spatial grid partitions the world into fixed-size cells.
// Each cell maintains a hash-set of actors currently in it.
// Insert / remove / move are all O(1) average, no heap allocation
// on move (we reuse the existing HashMap entry).

use crate::types::{ActorId, GridCell, Position, WorldConfig};
use ahash::AHashMap;
use std::collections::HashSet;

pub struct SpatialGrid {
    cells: AHashMap<GridCell, HashSet<ActorId>>,
    cell_size: f32,
}

impl SpatialGrid {
    pub fn new(cfg: &WorldConfig) -> Self {
        // Pre-allocate roughly 10% of cells to reduce rehashing.
        let cap = (cfg.grid_width * cfg.grid_height / 10) as usize;
        Self {
            cells: AHashMap::with_capacity(cap),
            cell_size: cfg.cell_size,
        }
    }

    #[inline]
    fn cell_of(&self, pos: Position) -> GridCell {
        pos.to_grid_cell(self.cell_size)
    }

    /// Insert an actor at position. Returns the grid cell it was placed in.
    #[inline]
    pub fn insert(&mut self, id: ActorId, pos: Position) -> GridCell {
        let cell = self.cell_of(pos);
        self.cells.entry(cell).or_default().insert(id);
        cell
    }

    /// Remove an actor from a known cell. O(1).
    #[inline]
    pub fn remove(&mut self, id: ActorId, cell: GridCell) -> bool {
        if let Some(set) = self.cells.get_mut(&cell) {
            let removed = set.remove(&id);
            if set.is_empty() {
                self.cells.remove(&cell);
            }
            return removed;
        }
        false
    }

    /// Move actor from old_cell to new position. Returns new cell.
    /// The caller is responsible for updating the actor's stored cell.
    #[inline]
    pub fn move_actor(&mut self, id: ActorId, old_cell: GridCell, new_pos: Position) -> GridCell {
        let new_cell = self.cell_of(new_pos);
        if old_cell == new_cell {
            return new_cell;
        }
        self.remove(id, old_cell);
        self.cells.entry(new_cell).or_default().insert(id);
        new_cell
    }

    /// Return all actor IDs in cells within Chebyshev distance `radius` of `center`.
    /// Allocates a Vec — caller should only call from non-hot paths.
    pub fn query_radius(&self, center: GridCell, radius: i32) -> Vec<ActorId> {
        let mut result = Vec::new();
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let cell = GridCell(center.0 + dx, center.1 + dy);
                if let Some(set) = self.cells.get(&cell) {
                    result.extend(set.iter().copied());
                }
            }
        }
        result
    }

    /// True if actor is in the cell it claims to be in.
    #[inline]
    pub fn contains(&self, id: ActorId, cell: GridCell) -> bool {
        self.cells.get(&cell).is_some_and(|s| s.contains(&id))
    }

    /// Count of occupied cells (for diagnostics).
    pub fn occupied_cells(&self) -> usize {
        self.cells.len()
    }

    /// Total actors across all cells.
    pub fn total_actors(&self) -> usize {
        self.cells.values().map(|s| s.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ActorId;

    fn default_grid() -> SpatialGrid {
        SpatialGrid::new(&WorldConfig::default())
    }

    #[test]
    fn insert_and_contains() {
        let mut g = default_grid();
        let id = ActorId(1);
        let pos = Position::new(5.0, 5.0);
        let cell = g.insert(id, pos);
        assert!(g.contains(id, cell));
        assert_eq!(cell, GridCell(0, 0));
    }

    #[test]
    fn remove_actor() {
        let mut g = default_grid();
        let id = ActorId(2);
        let cell = g.insert(id, Position::new(15.0, 15.0));
        assert!(g.remove(id, cell));
        assert!(!g.contains(id, cell));
        assert_eq!(g.occupied_cells(), 0);
    }

    #[test]
    fn move_to_different_cell() {
        let mut g = default_grid();
        let id = ActorId(3);
        let old_cell = g.insert(id, Position::new(5.0, 5.0));
        let new_cell = g.move_actor(id, old_cell, Position::new(55.0, 55.0));
        assert_ne!(old_cell, new_cell);
        assert!(!g.contains(id, old_cell));
        assert!(g.contains(id, new_cell));
    }

    #[test]
    fn move_within_same_cell_is_noop() {
        let mut g = default_grid();
        let id = ActorId(4);
        let cell = g.insert(id, Position::new(1.0, 1.0));
        let new_cell = g.move_actor(id, cell, Position::new(9.9, 9.9));
        assert_eq!(cell, new_cell);
        assert!(g.contains(id, cell));
    }

    #[test]
    fn query_radius_finds_neighbours() {
        let mut g = default_grid();
        // Insert actors in a 3×3 neighbourhood around (0,0)
        let mut ids = vec![];
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                let id = ActorId((dx + 10 + (dy + 10) * 100) as u64);
                let pos = Position::new(dx as f32 * 10.0 + 5.0, dy as f32 * 10.0 + 5.0);
                g.insert(id, pos);
                ids.push(id);
            }
        }
        let found = g.query_radius(GridCell(0, 0), 1);
        assert_eq!(found.len(), 9);
    }

    #[test]
    fn invariant_actor_count_after_mass_ops() {
        let mut g = default_grid();
        let mut cells = vec![];
        for i in 0..100 {
            let id = ActorId(i);
            let pos = Position::new((i as f32 % 10.0) * 15.0, (i as f32 / 10.0) * 15.0);
            cells.push((id, g.insert(id, pos)));
        }
        assert_eq!(g.total_actors(), 100);
        // Remove half
        for &(id, cell) in &cells[..50] {
            g.remove(id, cell);
        }
        assert_eq!(g.total_actors(), 50);
    }
}
