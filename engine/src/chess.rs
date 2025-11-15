use anyhow::{Error as E, Result};
use std::fmt;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct FenError(String);

impl FenError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FenError {}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Color {
    White = 0,
    Black,
}

impl Color {
    fn fen_active(self) -> char {
        match self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }

    fn opponent(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PieceKind {
    Pawn = 0,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl From<usize> for PieceKind {
    fn from(v: usize) -> Self {
        match v {
            0 => Self::Pawn,
            1 => Self::Knight,
            2 => Self::Bishop,
            3 => Self::Rook,
            4 => Self::Queen,
            5 => Self::King,
            _ => panic!("Cannot convert from {} to PieceKind.", v),
        }
    }
}

impl PieceKind {
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    pub const fn new(color: Color, kind: PieceKind) -> Self {
        Self { color, kind }
    }

    fn to_fen_symbol(self) -> char {
        let symbol = match self.kind {
            PieceKind::Pawn => 'p',
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::King => 'k',
        };

        match self.color {
            Color::White => symbol.to_ascii_uppercase(),
            Color::Black => symbol,
        }
    }

    fn betray(&self) -> Self {
        Self {
            color: self.color.opponent(),
            kind: self.kind,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Square {
    rank: u8,
    file: u8,
}

impl Square {
    pub fn new(rank: u8, file: u8) -> Option<Self> {
        if rank < 8 && file < 8 {
            Some(Self { rank, file })
        } else {
            None
        }
    }

    pub const fn unchecked(rank: u8, file: u8) -> Self {
        Self { rank, file }
    }

    pub fn rank(self) -> u8 {
        self.rank
    }

    pub fn file(self) -> u8 {
        self.file
    }

    fn to_indices(self) -> (usize, usize) {
        (self.rank as usize, self.file as usize)
    }

    fn to_index(self) -> usize {
        self.rank as usize * 8 + self.file as usize
    }

    pub fn to_algebraic(self) -> String {
        let rank_char = (b'1' + self.rank) as char;
        let file_char = (b'a' + self.file) as char;
        format!("{file_char}{rank_char}")
    }

    pub fn from_algebraic(value: &str) -> Option<Self> {
        let bytes = value.as_bytes();
        if bytes.len() != 2 {
            return None;
        }

        let file = bytes[0];
        let rank = bytes[1];

        if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
            return None;
        }

        let rank_idx = rank - b'1';
        let file_idx = file - b'a';
        Some(Self {
            rank: rank_idx,
            file: file_idx,
        })
    }

    pub fn offset(self, dr: i8, df: i8) -> Option<Self> {
        let rank = self.rank as i8 + dr;
        let file = self.file as i8 + df;
        if (0..=7).contains(&rank) && (0..=7).contains(&file) {
            Some(Square::unchecked(rank as u8, file as u8))
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_algebraic())
    }
}

fn zobrist_keys() -> &'static ZobristKeys {
    static KEYS: OnceLock<ZobristKeys> = OnceLock::new();
    KEYS.get_or_init(ZobristKeys::new)
}

struct ZobristKeys {
    pieces: [[[u64; 64]; 6]; 2],
    side_to_move: u64,
    castling: [u64; 4],
    en_passant: [u64; 8],
}

impl ZobristKeys {
    fn new() -> Self {
        let mut rng = SplitMix64::new(0xDEAD_BEEF_F00D_BAAD);
        let mut pieces = [[[0_u64; 64]; 6]; 2];
        for color in 0..2 {
            for kind in 0..6 {
                for square in 0..64 {
                    pieces[color][kind][square] = rng.next_u64();
                }
            }
        }

        let mut castling = [0_u64; 4];
        for entry in &mut castling {
            *entry = rng.next_u64();
        }

        let mut en_passant = [0_u64; 8];
        for entry in &mut en_passant {
            *entry = rng.next_u64();
        }

        let side_to_move = rng.next_u64();

        Self {
            pieces,
            side_to_move,
            castling,
            en_passant,
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_add(0x9E37_79B9_7F4A_7C15_u64);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Move {
    pub from: Square,
    pub to: Square,
}

impl Move {
    pub const fn new(from: Square, to: Square) -> Self {
        Self { from, to }
    }

    pub fn to_uci(self) -> String {
        format!("{}{}", self.from.to_algebraic(), self.to.to_algebraic())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
}

impl CastlingRights {
    pub fn as_fen(self) -> String {
        let mut buffer = String::with_capacity(4);
        if self.white_kingside {
            buffer.push('K');
        }
        if self.white_queenside {
            buffer.push('Q');
        }
        if self.black_kingside {
            buffer.push('k');
        }
        if self.black_queenside {
            buffer.push('q');
        }

        if buffer.is_empty() {
            "-".to_string()
        } else {
            buffer
        }
    }

    fn validates_for(self, board: &Board) -> bool {
        // Rely on classical chess: castling requires king and rooks to exist on home squares.
        let white_king_ok =
            board.has_piece(Square::unchecked(0, 4), PieceKind::King, Color::White);
        let black_king_ok =
            board.has_piece(Square::unchecked(7, 4), PieceKind::King, Color::Black);

        if self.white_kingside
            && !(white_king_ok
                && board.has_piece(
                    Square::unchecked(0, 7),
                    PieceKind::Rook,
                    Color::White,
                ))
        {
            return false;
        }
        if self.white_queenside
            && !(white_king_ok
                && board.has_piece(
                    Square::unchecked(0, 0),
                    PieceKind::Rook,
                    Color::White,
                ))
        {
            return false;
        }
        if self.black_kingside
            && !(black_king_ok
                && board.has_piece(
                    Square::unchecked(7, 7),
                    PieceKind::Rook,
                    Color::Black,
                ))
        {
            return false;
        }
        if self.black_queenside
            && !(black_king_ok
                && board.has_piece(
                    Square::unchecked(7, 0),
                    PieceKind::Rook,
                    Color::Black,
                ))
        {
            return false;
        }

        true
    }
}

#[derive(Clone)]
pub struct Board {
    squares: [[Option<Piece>; 8]; 8],
    pub active_color: Color,
    pub castling_rights: CastlingRights,
    pub en_passant: Option<Square>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    zobrist: u64,
}

impl Default for Board {
    fn default() -> Self {
        Self::starting_position()
    }
}

impl Board {
    pub fn empty() -> Self {
        Self {
            squares: [[None; 8]; 8],
            active_color: Color::White,
            castling_rights: CastlingRights::default(),
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            zobrist: 0,
        }
    }

    pub fn starting_position() -> Self {
        use Color::{Black, White};
        use PieceKind::*;

        let mut board = Board::empty();

        let back_rank = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook];

        for (file, kind) in back_rank.into_iter().enumerate() {
            board.set_piece(
                Square::unchecked(0, file as u8),
                Some(Piece::new(White, kind)),
            );
            board.set_piece(
                Square::unchecked(1, file as u8),
                Some(Piece::new(White, Pawn)),
            );
            board.set_piece(
                Square::unchecked(6, file as u8),
                Some(Piece::new(Black, Pawn)),
            );
            board.set_piece(
                Square::unchecked(7, file as u8),
                Some(Piece::new(Black, kind)),
            );
        }

        board.set_castling_rights(CastlingRights {
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
        });
        board
    }

    pub fn set_piece(&mut self, square: Square, piece: Option<Piece>) {
        let (r, f) = square.to_indices();
        let idx = square.to_index();
        let keys = zobrist_keys();
        if let Some(old) = self.squares[r][f] {
            self.zobrist ^= keys.pieces[old.color.idx()][old.kind.idx()][idx];
        }
        self.squares[r][f] = piece;
        if let Some(new_piece) = self.squares[r][f] {
            self.zobrist ^=
                keys.pieces[new_piece.color.idx()][new_piece.kind.idx()][idx];
        }
    }

    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        let (r, f) = square.to_indices();
        self.squares[r][f]
    }

    pub fn set_active_color(&mut self, color: Color) {
        if self.active_color != color {
            self.zobrist ^= zobrist_keys().side_to_move;
            self.active_color = color;
        }
    }

    pub fn set_en_passant(&mut self, square: Option<Square>) {
        let keys = zobrist_keys();
        if let Some(old) = self.en_passant {
            self.zobrist ^= keys.en_passant[old.file() as usize];
        }
        self.en_passant = square;
        if let Some(new_sq) = self.en_passant {
            self.zobrist ^= keys.en_passant[new_sq.file() as usize];
        }
    }

    pub fn set_castling_rights(&mut self, rights: CastlingRights) {
        let old = self.castling_rights;
        self.castling_rights = rights;
        self.update_castling_hash(old, rights);
    }

    fn toggle_active_color(&mut self) {
        self.zobrist ^= zobrist_keys().side_to_move;
        self.active_color = self.active_color.opponent();
    }

    fn update_castling_hash(&mut self, old: CastlingRights, new: CastlingRights) {
        let keys = zobrist_keys();
        if old.white_kingside != new.white_kingside {
            self.zobrist ^= keys.castling[0];
        }
        if old.white_queenside != new.white_queenside {
            self.zobrist ^= keys.castling[1];
        }
        if old.black_kingside != new.black_kingside {
            self.zobrist ^= keys.castling[2];
        }
        if old.black_queenside != new.black_queenside {
            self.zobrist ^= keys.castling[3];
        }
    }

    pub fn play_move(&mut self, mv: Move) -> bool {
        if self.move_is_legal(mv) {
            self.apply_move_unchecked(mv);
            true
        } else {
            false
        }
    }

    fn has_piece(&self, square: Square, kind: PieceKind, color: Color) -> bool {
        self.piece_at(square) == Some(Piece::new(color, kind))
    }

    pub fn hash(&self) -> u64 {
        self.zobrist
    }

    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let mut parts = fen.split_whitespace();
        let placement = parts
            .next()
            .ok_or_else(|| FenError::new("FEN missing piece placement field"))?;
        let active = parts
            .next()
            .ok_or_else(|| FenError::new("FEN missing active color field"))?;
        let castling = parts
            .next()
            .ok_or_else(|| FenError::new("FEN missing castling rights field"))?;
        let en_passant = parts
            .next()
            .ok_or_else(|| FenError::new("FEN missing en passant field"))?;
        let halfmove_field = parts.next();
        let fullmove_field = parts.next();
        if parts.next().is_some() {
            return Err(FenError::new("FEN has extra fields"));
        }

        let mut board = Board::empty();
        Self::parse_placement(placement, &mut board)?;
        let active_color = match active {
            "w" => Color::White,
            "b" => Color::Black,
            value => {
                return Err(FenError::new(format!("invalid active color '{value}'")));
            }
        };
        board.set_active_color(active_color);

        let castling_rights = Self::parse_castling(castling)?;
        board.set_castling_rights(castling_rights);
        let ep_square = if en_passant == "-" {
            None
        } else {
            Some(Square::from_algebraic(en_passant).ok_or_else(|| {
                FenError::new(format!("invalid en passant square '{en_passant}'"))
            })?)
        };
        board.set_en_passant(ep_square);

        board.halfmove_clock = match halfmove_field {
            Some(value) => value
                .parse::<u32>()
                .map_err(|_| FenError::new("invalid halfmove clock"))?,
            None => 0,
        };
        board.fullmove_number = match fullmove_field {
            Some(value) => value
                .parse::<u32>()
                .map_err(|_| FenError::new("invalid fullmove number"))?,
            None => 1,
        };
        if board.fullmove_number == 0 {
            return Err(FenError::new("fullmove number must be > 0"));
        }

        Ok(board)
    }

    fn parse_castling(field: &str) -> Result<CastlingRights, FenError> {
        if field == "-" {
            return Ok(CastlingRights::default());
        }

        let mut rights = CastlingRights::default();
        for ch in field.chars() {
            match ch {
                'K' => rights.white_kingside = true,
                'Q' => rights.white_queenside = true,
                'k' => rights.black_kingside = true,
                'q' => rights.black_queenside = true,
                _ => {
                    return Err(FenError::new(format!(
                        "invalid castling flag '{ch}'"
                    )));
                }
            }
        }

        Ok(rights)
    }

    fn parse_placement(field: &str, board: &mut Board) -> Result<(), FenError> {
        let ranks: Vec<&str> = field.split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError::new("piece placement must have 8 ranks"));
        }

        for (rank_idx, rank_data) in ranks.iter().rev().enumerate() {
            let mut file = 0;
            for ch in rank_data.chars() {
                if ch.is_ascii_digit() {
                    let empty = ch.to_digit(10).unwrap();
                    file += empty as usize;
                } else {
                    let piece = Self::piece_from_fen(ch).ok_or_else(|| {
                        FenError::new(format!("invalid piece char '{ch}'"))
                    })?;
                    if file >= 8 {
                        return Err(FenError::new("too many squares in rank"));
                    }
                    board.set_piece(
                        Square::unchecked(rank_idx as u8, file as u8),
                        Some(piece),
                    );
                    file += 1;
                }
            }

            if file != 8 {
                return Err(FenError::new("each rank must have 8 squares"));
            }
        }

        Ok(())
    }

    fn piece_from_fen(ch: char) -> Option<Piece> {
        let color = if ch.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let kind = match ch.to_ascii_lowercase() {
            'p' => PieceKind::Pawn,
            'n' => PieceKind::Knight,
            'b' => PieceKind::Bishop,
            'r' => PieceKind::Rook,
            'q' => PieceKind::Queen,
            'k' => PieceKind::King,
            _ => return None,
        };

        Some(Piece::new(color, kind))
    }

    pub fn fen(&self) -> String {
        let mut result = String::with_capacity(80);

        for rank in (0..8).rev() {
            let mut empty = 0;

            for file in 0..8 {
                match self.squares[rank][file] {
                    Some(piece) => {
                        if empty > 0 {
                            result.push(char::from_digit(empty, 10).unwrap());
                            empty = 0;
                        }
                        result.push(piece.to_fen_symbol());
                    }
                    None => empty += 1,
                }
            }

            if empty > 0 {
                result.push(char::from_digit(empty, 10).unwrap());
            }

            if rank != 0 {
                result.push('/');
            }
        }

        result.push(' ');
        result.push(self.active_color.fen_active());
        result.push(' ');
        result.push_str(&self.castling_rights.as_fen());
        result.push(' ');

        match self.en_passant {
            Some(sq) => result.push_str(&sq.to_algebraic()),
            None => result.push('-'),
        }

        result.push(' ');
        result.push_str(&self.halfmove_clock.to_string());
        result.push(' ');
        result.push_str(&self.fullmove_number.to_string());

        result
    }

    pub fn is_legal(&self) -> bool {
        let mut white_kings = 0;
        let mut black_kings = 0;

        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.squares[rank][file] {
                    match piece.kind {
                        PieceKind::King => match piece.color {
                            Color::White => white_kings += 1,
                            Color::Black => black_kings += 1,
                        },
                        PieceKind::Pawn => {
                            // Pawns cannot be on the first or eighth rank.
                            if rank == 0 || rank == 7 {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if white_kings != 1 || black_kings != 1 {
            return false;
        }

        if !self.castling_rights.validates_for(self) {
            return false;
        }

        match self.en_passant {
            Some(square) => matches!(square.rank, 2 | 5),
            None => true,
        }
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        for mv in self.pseudo_legal_moves() {
            if self.move_is_legal(mv) {
                moves.push(mv);
            }
        }
        moves
    }

    fn pseudo_legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.squares[rank][file]
                    && piece.color == self.active_color
                {
                    let square = Square::unchecked(rank as u8, file as u8);
                    self.generate_moves_for_piece(square, piece, &mut moves);
                }
            }
        }
        moves
    }

    fn generate_moves_for_piece(
        &self,
        square: Square,
        piece: Piece,
        moves: &mut Vec<Move>,
    ) {
        match piece.kind {
            PieceKind::Pawn => self.generate_pawn_moves(square, piece.color, moves),
            PieceKind::Knight => {
                self.generate_knight_moves(square, piece.color, moves)
            }
            PieceKind::Bishop => self.generate_sliding_moves(
                square,
                piece.color,
                moves,
                &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
            ),
            PieceKind::Rook => self.generate_sliding_moves(
                square,
                piece.color,
                moves,
                &[(1, 0), (-1, 0), (0, 1), (0, -1)],
            ),
            PieceKind::Queen => self.generate_sliding_moves(
                square,
                piece.color,
                moves,
                &[
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ],
            ),
            PieceKind::King => self.generate_king_moves(square, piece.color, moves),
        }
    }

    fn generate_pawn_moves(
        &self,
        square: Square,
        color: Color,
        moves: &mut Vec<Move>,
    ) {
        let dir: i8 = match color {
            Color::White => 1,
            Color::Black => -1,
        };

        if let Some(one_step) = square.offset(dir, 0)
            && self.piece_at(one_step).is_none()
        {
            moves.push(Move::new(square, one_step));

            let start_rank = match color {
                Color::White => 1,
                Color::Black => 6,
            };
            if square.rank() == start_rank
                && let Some(two_step) = square.offset(dir * 2, 0)
                && self.piece_at(two_step).is_none()
            {
                moves.push(Move::new(square, two_step));
            }
        }

        for df in [-1, 1] {
            if let Some(target) = square.offset(dir, df)
                && let Some(piece) = self.piece_at(target)
                && piece.color != color
            {
                moves.push(Move::new(square, target));
            }
        }
    }

    fn generate_knight_moves(
        &self,
        square: Square,
        color: Color,
        moves: &mut Vec<Move>,
    ) {
        const KNIGHT_OFFSETS: [(i8, i8); 8] = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ];
        for (dr, df) in KNIGHT_OFFSETS {
            if let Some(target) = square.offset(dr, df) {
                match self.piece_at(target) {
                    None => moves.push(Move::new(square, target)),
                    Some(piece) if piece.color != color => {
                        moves.push(Move::new(square, target))
                    }
                    _ => {}
                }
            }
        }
    }

    fn generate_sliding_moves(
        &self,
        square: Square,
        color: Color,
        moves: &mut Vec<Move>,
        directions: &[(i8, i8)],
    ) {
        for (dr, df) in directions {
            let mut current = square;
            while let Some(next) = current.offset(*dr, *df) {
                match self.piece_at(next) {
                    None => moves.push(Move::new(square, next)),
                    Some(piece) if piece.color != color => {
                        moves.push(Move::new(square, next));
                        break;
                    }
                    _ => break,
                }
                current = next;
            }
        }
    }

    fn generate_king_moves(
        &self,
        square: Square,
        color: Color,
        moves: &mut Vec<Move>,
    ) {
        for df in -1..=1 {
            for dr in -1..=1 {
                if df == 0 && dr == 0 {
                    continue;
                }

                if let Some(target) = square.offset(dr, df) {
                    match self.piece_at(target) {
                        None => moves.push(Move::new(square, target)),
                        Some(piece) if piece.color != color => {
                            moves.push(Move::new(square, target))
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn move_is_legal(&self, mv: Move) -> bool {
        let piece = match self.piece_at(mv.from) {
            Some(piece) => piece,
            None => return false,
        };

        if piece.color != self.active_color {
            return false;
        }

        if let Some(dest_piece) = self.piece_at(mv.to)
            && dest_piece.color == piece.color
        {
            return false;
        }

        let mut next = self.clone();
        next.apply_move_unchecked(mv);
        !next.is_in_check(self.active_color)
    }

    pub(crate) fn apply_move_unchecked(&mut self, mv: Move) {
        let piece = self.piece_at(mv.from).expect("move must have a piece");
        let captured = self.piece_at(mv.to);

        self.update_castling_after_move(mv, piece, captured);

        self.set_piece(mv.from, None);

        let mut moved_piece = piece;
        if moved_piece.kind == PieceKind::Pawn
            && (mv.to.rank() == 0 || mv.to.rank() == 7)
        {
            moved_piece = Piece::new(moved_piece.color, PieceKind::Queen);
        }

        self.set_piece(mv.to, Some(moved_piece));
        self.set_en_passant(None);
        self.halfmove_clock =
            if moved_piece.kind == PieceKind::Pawn || captured.is_some() {
                0
            } else {
                self.halfmove_clock.saturating_add(1)
            };

        if self.active_color == Color::Black {
            self.fullmove_number += 1;
        }

        self.toggle_active_color();
    }

    fn update_castling_after_move(
        &mut self,
        mv: Move,
        moving: Piece,
        captured: Option<Piece>,
    ) {
        let mut rights = self.castling_rights;
        if moving.kind == PieceKind::King {
            match moving.color {
                Color::White => {
                    rights.white_kingside = false;
                    rights.white_queenside = false;
                }
                Color::Black => {
                    rights.black_kingside = false;
                    rights.black_queenside = false;
                }
            }
        }

        if moving.kind == PieceKind::Rook {
            Self::disable_rook_castling(&mut rights, mv.from, moving.color);
        }

        if let Some(captured_piece) = captured
            && captured_piece.kind == PieceKind::Rook
        {
            Self::disable_rook_castling(&mut rights, mv.to, captured_piece.color);
        }

        if rights != self.castling_rights {
            self.set_castling_rights(rights);
        }
    }

    fn disable_rook_castling(
        rights: &mut CastlingRights,
        square: Square,
        color: Color,
    ) {
        match (color, square.file(), square.rank()) {
            (Color::White, 0, 0) => rights.white_queenside = false,
            (Color::White, 7, 0) => rights.white_kingside = false,
            (Color::Black, 0, 7) => rights.black_queenside = false,
            (Color::Black, 7, 7) => rights.black_kingside = false,
            _ => {}
        }
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        if let Some(king_square) = self.king_square(color) {
            self.square_attacked(king_square, color.opponent())
        } else {
            false
        }
    }

    fn king_square(&self, color: Color) -> Option<Square> {
        for rank in 0..8 {
            for file in 0..8 {
                if self.squares[rank][file]
                    == Some(Piece::new(color, PieceKind::King))
                {
                    return Some(Square::unchecked(rank as u8, file as u8));
                }
            }
        }
        None
    }

    fn square_attacked(&self, square: Square, by: Color) -> bool {
        let pawn_dirs: [(i8, i8); 2] = match by {
            Color::White => [(-1, -1), (-1, 1)],
            Color::Black => [(1, -1), (1, 1)],
        };
        for (dr, df) in pawn_dirs {
            if let Some(source) = square.offset(dr, df)
                && self.piece_at(source) == Some(Piece::new(by, PieceKind::Pawn))
            {
                return true;
            }
        }

        const KNIGHT_OFFSETS: [(i8, i8); 8] = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ];
        for (dr, df) in KNIGHT_OFFSETS {
            if let Some(source) = square.offset(dr, df)
                && self.piece_at(source) == Some(Piece::new(by, PieceKind::Knight))
            {
                return true;
            }
        }

        for dr in -1..=1 {
            for df in -1..=1 {
                if dr == 0 && df == 0 {
                    continue;
                }
                if let Some(source) = square.offset(dr, df)
                    && self.piece_at(source) == Some(Piece::new(by, PieceKind::King))
                {
                    return true;
                }
            }
        }

        if self.attack_in_directions(
            square,
            by,
            &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
            &[PieceKind::Bishop, PieceKind::Queen],
        ) {
            return true;
        }
        if self.attack_in_directions(
            square,
            by,
            &[(1, 0), (-1, 0), (0, 1), (0, -1)],
            &[PieceKind::Rook, PieceKind::Queen],
        ) {
            return true;
        }

        false
    }

    fn attack_in_directions(
        &self,
        square: Square,
        color: Color,
        directions: &[(i8, i8)],
        attackers: &[PieceKind],
    ) -> bool {
        for (dr, df) in directions {
            let mut current = square;
            while let Some(next) = current.offset(*dr, *df) {
                if let Some(piece) = self.piece_at(next) {
                    if piece.color == color && attackers.contains(&piece.kind) {
                        return true;
                    }
                    break;
                }
                current = next;
            }
        }
        false
    }

    pub fn halfka(&self) -> Result<Vec<f32>> {
        if self.active_color == Color::White {
            self.halfka_from_white()
        } else {
            let mut flipped = self.flipped();
            flipped.set_active_color(Color::White);
            flipped.halfka_from_white()
        }
    }

    fn flipped(&self) -> Self {
        let mut flipped = self.clone();
        for rank in 0..4 {
            for file in 0..8 {
                let piece = self
                    .piece_at(Square::unchecked(rank, file))
                    .map(|v| v.betray());
                flipped.set_piece(
                    Square::unchecked(rank, file),
                    self.piece_at(Square::unchecked(7 - rank, file))
                        .map(|v| v.betray()),
                );
                flipped.set_piece(Square::unchecked(7 - rank, file), piece);
            }
        }
        flipped
    }

    fn halfka_from_white(&self) -> Result<Vec<f32>> {
        let mut features = vec![0_f32; 64 * (12 * 64)];
        let mut king_square = None;

        let square_index = |rank: usize, file: usize| rank * 8 + file;

        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.squares[rank][file]
                    && piece.kind == PieceKind::King
                    && piece.color == self.active_color
                {
                    if king_square.is_none() {
                        king_square = Some((rank, file));
                    } else {
                        return Err(E::msg("Multiple king has found."));
                    }
                }
            }
        }
        let king_square = if let Some(ks) = king_square {
            ks
        } else {
            return Err(E::msg("King is missing."));
        };

        let get_index = |rank: u32, file: u32| {
            if let Some(piece) = self.squares[rank as usize][file as usize] {
                let mut idx =
                    square_index(king_square.0, king_square.1) as u32 * (12 * 64);
                let piecekind = if piece.color == self.active_color {
                    piece.kind as u32
                } else {
                    piece.kind as u32 + 6
                };
                idx += piecekind * 64
                    + square_index(rank as usize, file as usize) as u32;
                Some(idx as usize)
            } else {
                None
            }
        };
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(idx) = get_index(rank, file) {
                    features[idx] = 1_f32;
                }
            }
        }

        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_fen_matches_standard() {
        let board = Board::starting_position();
        assert_eq!(
            board.fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    #[test]
    fn empty_board_is_illegal_due_to_missing_kings() {
        let board = Board::empty();
        assert!(!board.is_legal());
    }

    #[test]
    fn pawn_on_back_rank_is_illegal() {
        let mut board = Board::empty();
        board.set_piece(
            Square::unchecked(4, 4),
            Some(Piece::new(Color::White, PieceKind::King)),
        );
        board.set_piece(
            Square::unchecked(4, 3),
            Some(Piece::new(Color::Black, PieceKind::King)),
        );
        board.set_piece(
            Square::unchecked(0, 0),
            Some(Piece::new(Color::White, PieceKind::Pawn)),
        );

        assert!(!board.is_legal());
    }

    #[test]
    fn castling_rights_require_rooks() {
        let mut board = Board::starting_position();
        board.set_piece(Square::unchecked(7, 7), None);
        assert!(!board.is_legal());
    }

    #[test]
    fn en_passant_square_must_be_on_third_or_sixth_rank() {
        let mut board = Board::starting_position();
        board.set_en_passant(Some(Square::unchecked(0, 0)));
        assert!(!board.is_legal());

        board.set_en_passant(Some(Square::unchecked(2, 0)));
        assert!(board.is_legal());
    }

    #[test]
    fn square_new_bounds_check() {
        assert!(Square::new(0, 0).is_some());
        assert!(Square::new(7, 7).is_some());
        assert!(Square::new(8, 0).is_none());
        assert!(Square::new(0, 8).is_none());
    }

    #[test]
    fn starting_fen_round_trip() {
        let start = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        .expect("valid FEN");
        assert_eq!(
            start.fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    #[test]
    fn truncated_fen_defaults_missing_clocks() {
        let board = Board::from_fen("8/8/8/8/8/8/8/8 w - -")
            .expect("missing clocks should succeed");
        assert_eq!(board.halfmove_clock, 0);
        assert_eq!(board.fullmove_number, 1);

        let board = Board::from_fen("8/8/8/8/8/8/8/8 b - - 12")
            .expect("missing fullmove number should default to 1");
        assert_eq!(board.halfmove_clock, 12);
        assert_eq!(board.fullmove_number, 1);
    }

    #[test]
    fn invalid_fen_reports_error() {
        assert!(Board::from_fen("invalid fen string").is_err());
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 0").is_err());
    }

    #[test]
    fn starting_position_has_twenty_legal_moves() {
        let board = Board::starting_position();
        assert_eq!(board.legal_moves().len(), 20);
    }

    #[test]
    fn pinned_piece_cannot_move() {
        let mut board = Board::empty();
        board.set_active_color(Color::White);
        board.set_piece(
            Square::unchecked(7, 4),
            Some(Piece::new(Color::White, PieceKind::King)),
        );
        board.set_piece(
            Square::unchecked(0, 0),
            Some(Piece::new(Color::Black, PieceKind::King)),
        );
        board.set_piece(
            Square::unchecked(6, 4),
            Some(Piece::new(Color::White, PieceKind::Rook)),
        );
        board.set_piece(
            Square::unchecked(0, 4),
            Some(Piece::new(Color::Black, PieceKind::Rook)),
        );

        let moves = board.legal_moves();
        let illegal = Move::new(Square::unchecked(6, 4), Square::unchecked(6, 5));
        assert!(
            !moves.contains(&illegal),
            "Pinned rook should not be allowed to move"
        );
    }

    #[test]
    fn halfka_starting_position() {
        let board = Board::starting_position();
        let features = board.halfka().expect("halfka should succeed");

        assert_eq!(features.len(), 64 * 12 * 64);
        let ones = features.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(
            ones, 32,
            "starting position should have 32 feature activations"
        );

        let king_sq = (0_u8, 4_u8); // e1
        let idx = |piece_sq: (u8, u8), kind: PieceKind, friendly: bool| -> usize {
            let king_index = king_sq.0 as usize * 8 + king_sq.1 as usize;
            let plane_offset = king_index * (12 * 64);
            let square_index = piece_sq.0 as usize * 8 + piece_sq.1 as usize;
            let kind_index = kind as usize + if friendly { 0 } else { 6 };
            plane_offset + kind_index * 64 + square_index
        };

        // White pawn on a2
        assert_eq!(features[idx((1, 0), PieceKind::Pawn, true)], 1.0);
        // Black pawn on a7
        assert_eq!(features[idx((6, 0), PieceKind::Pawn, false)], 1.0);
        // White king on e1
        assert_eq!(features[idx((0, 4), PieceKind::King, true)], 1.0);
        // Black king on e8
        assert_eq!(features[idx((7, 4), PieceKind::King, false)], 1.0);
    }

    #[test]
    fn halfka_complex_position_1() {
        let mut board = Board::empty();
        board.set_active_color(Color::White);
        board.set_piece(
            Square::unchecked(0, 0),
            Some(Piece::new(Color::White, PieceKind::King)),
        );
        board.set_piece(
            Square::unchecked(0, 1),
            Some(Piece::new(Color::White, PieceKind::Knight)),
        );
        board.set_piece(
            Square::unchecked(7, 7),
            Some(Piece::new(Color::Black, PieceKind::Rook)),
        );

        let features = board.halfka().expect("halfka should succeed");
        let king_sq = (0_u8, 0_u8); // a1
        let idx = |piece_sq: (u8, u8), kind: PieceKind, friendly: bool| -> usize {
            let king_index = king_sq.0 as usize * 8 + king_sq.1 as usize;
            let plane_offset = king_index * (12 * 64);
            let square_index = piece_sq.0 as usize * 8 + piece_sq.1 as usize;
            let kind_index = kind as usize + if friendly { 0 } else { 6 };
            plane_offset + kind_index * 64 + square_index
        };

        assert_eq!(features[idx((0, 0), PieceKind::King, true)], 1.0);
        assert_eq!(features[idx((0, 1), PieceKind::Knight, true)], 1.0);
        assert_eq!(features[idx((7, 7), PieceKind::Rook, false)], 1.0);

        let ones = features.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(ones, 3);
    }

    #[test]
    fn halfka_complex_position_2() {
        let mut board = Board::empty();
        board.set_active_color(Color::Black);
        board.set_piece(
            Square::unchecked(7, 7),
            Some(Piece::new(Color::Black, PieceKind::King)),
        );
        board.set_piece(
            Square::unchecked(6, 7),
            Some(Piece::new(Color::Black, PieceKind::Pawn)),
        );
        board.set_piece(
            Square::unchecked(0, 0),
            Some(Piece::new(Color::White, PieceKind::Bishop)),
        );

        let features = board.halfka().expect("halfka should succeed");

        let king_sq = (0_u8, 7_u8); // mirrored h8 after flip
        let idx = |piece_sq: (u8, u8), kind: PieceKind, friendly: bool| -> usize {
            let king_index = king_sq.0 as usize * 8 + king_sq.1 as usize;
            let plane_offset = king_index * (12 * 64);
            let square_index = piece_sq.0 as usize * 8 + piece_sq.1 as usize;
            let kind_index = kind as usize + if friendly { 0 } else { 6 };
            plane_offset + kind_index * 64 + square_index
        };

        // Black king becomes white king on h1 after flip.
        assert_eq!(features[idx((0, 7), PieceKind::King, true)], 1.0);
        // Black pawn becomes white pawn on h2 after flip.
        assert_eq!(features[idx((1, 7), PieceKind::Pawn, true)], 1.0);
        // White bishop becomes black bishop on a8 after flip.
        assert_eq!(features[idx((7, 0), PieceKind::Bishop, false)], 1.0);

        let ones = features.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(ones, 3);
    }

    #[test]
    fn halfka_complex_position_3() {
        let mut board = Board::empty();
        board.set_active_color(Color::White);
        board.set_piece(
            Square::unchecked(7, 7),
            Some(Piece::new(Color::Black, PieceKind::King)),
        );

        assert!(
            board.halfka().is_err(),
            "missing active-color king should error"
        );
    }
}
