// Interest management: for each session (client connection), compute which
// actors are within its view radius and should receive state updates.
//
// Design: we track each session's "anchor" actor and its current grid cell.
// On each tick we query the spatial grid for all actors within radius R
// (Chebyshev distance) of that cell. Only those actors are included in
// the state delta sent to this client.

use crate::spatial_grid::SpatialGrid;
use crate::types::{ActorId, GridCell, SessionId};
use ahash::AHashMap;

pub struct InterestManager {
    /// Maps session → (anchor actor, current grid cell of anchor).
    session_anchors: AHashMap<SessionId, (ActorId, GridCell)>,
    /// View radius in grid cells (Chebyshev).
    radius: i32,
}

impl InterestManager {
    pub fn new(radius: i32) -> Self {
        Self {
            session_anchors: AHashMap::new(),
            radius,
        }
    }

    /// Register a session with its anchor actor and initial cell.
    pub fn register(&mut self, session: SessionId, anchor: ActorId, cell: GridCell) {
        self.session_anchors.insert(session, (anchor, cell));
    }

    /// Unregister a session (on disconnect).
    pub fn unregister(&mut self, session: SessionId) {
        self.session_anchors.remove(&session);
    }

    /// Update a session's anchor cell (call when anchor actor moves to new cell).
    #[inline]
    pub fn update_cell(&mut self, session: SessionId, new_cell: GridCell) {
        if let Some((_, cell)) = self.session_anchors.get_mut(&session) {
            *cell = new_cell;
        }
    }

    /// Return the set of ActorIds visible to this session, via the spatial grid.
    /// Allocates a Vec — only call from tick/broadcast path, not hot inner loops.
    #[inline]
    pub fn visible_actors(&self, session: SessionId, grid: &SpatialGrid) -> Vec<ActorId> {
        match self.session_anchors.get(&session) {
            None => vec![],
            Some(&(_, cell)) => grid.query_radius(cell, self.radius),
        }
    }

    /// Iterate all sessions with their anchor cells.
    pub fn sessions(&self) -> impl Iterator<Item = (SessionId, ActorId, GridCell)> + '_ {
        self.session_anchors
            .iter()
            .map(|(&sid, &(aid, cell))| (sid, aid, cell))
    }

    pub fn session_count(&self) -> usize {
        self.session_anchors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial_grid::SpatialGrid;
    use crate::types::{ActorId, Position, WorldConfig};

    fn setup() -> (SpatialGrid, InterestManager) {
        let cfg = WorldConfig::default();
        let grid = SpatialGrid::new(&cfg);
        let im = InterestManager::new(cfg.interest_radius);
        (grid, im)
    }

    #[test]
    fn no_actors_returns_empty() {
        let (grid, mut im) = setup();
        let sid = SessionId(1);
        im.register(sid, ActorId(1), GridCell(0, 0));
        assert!(im.visible_actors(sid, &grid).is_empty());
    }

    #[test]
    fn actors_in_radius_are_visible() {
        let (mut grid, mut im) = setup();
        let sid = SessionId(1);
        let anchor = ActorId(1);
        im.register(sid, anchor, GridCell(5, 5));

        // Place actors at various cells
        let near = ActorId(2);
        let far = ActorId(3);
        grid.insert(near, Position::new(55.0, 55.0)); // cell (5,5) — same cell
        grid.insert(far, Position::new(150.0, 150.0)); // cell (15,15) — out of radius=5

        let visible = im.visible_actors(sid, &grid);
        assert!(visible.contains(&near), "nearby actor should be visible");
        assert!(!visible.contains(&far), "far actor must not leak into view");
    }

    #[test]
    fn update_cell_moves_interest_window() {
        let (mut grid, mut im) = setup();
        let sid = SessionId(2);
        im.register(sid, ActorId(10), GridCell(0, 0));

        let actor_far = ActorId(20);
        grid.insert(actor_far, Position::new(150.0, 150.0)); // cell (15,15)

        // Initially not visible
        assert!(!im.visible_actors(sid, &grid).contains(&actor_far));

        // Move session anchor toward that actor
        im.update_cell(sid, GridCell(15, 15));
        assert!(im.visible_actors(sid, &grid).contains(&actor_far));
    }

    #[test]
    fn unregister_removes_session() {
        let (grid, mut im) = setup();
        let sid = SessionId(3);
        im.register(sid, ActorId(1), GridCell(0, 0));
        assert_eq!(im.session_count(), 1);
        im.unregister(sid);
        assert_eq!(im.session_count(), 0);
        // Querying unregistered session is safe and returns empty
        assert!(im.visible_actors(sid, &grid).is_empty());
    }

    #[test]
    fn no_leakage_invariant_random() {
        // Place actors randomly and assert that for any session with radius R,
        // no actor outside R is in the visible set.
        let cfg = WorldConfig {
            interest_radius: 3,
            ..Default::default()
        };
        let mut grid = SpatialGrid::new(&cfg);
        let mut im = InterestManager::new(cfg.interest_radius);

        let anchor_cell = GridCell(10, 10);
        let sid = SessionId(99);
        im.register(sid, ActorId(0), anchor_cell);

        // Actors strictly outside radius
        for i in 1..=20u64 {
            let offset = cfg.interest_radius + 2 + (i as i32 % 5);
            let pos = Position::new(
                (anchor_cell.0 + offset) as f32 * cfg.cell_size + 1.0,
                anchor_cell.1 as f32 * cfg.cell_size + 1.0,
            );
            grid.insert(ActorId(i), pos);
        }

        let visible = im.visible_actors(sid, &grid);
        assert!(
            visible.is_empty(),
            "No actors outside radius should be visible; got {:?}",
            visible
        );
    }
}
