class OpenTTDLEBridge extends AIController {
    function Start();
    function ChooseTownPair();
    function FindBuildableTileNear(center_tile);
    function BuildRoadBurst(start_tile, label);
    function TryRoad(tile_a, tile_b);
}

function OpenTTDLEBridge::Start()
{
    this.Sleep(1);
    AICompany.SetName("TycoonLE OpenTTD Speedrun");
    AILog.Info("OpenTTDLEBridge speedrun started.");

    local pair = this.ChooseTownPair();
    if (pair == null) {
        AILog.Error("Not enough towns to choose a visible objective.");
        while (true) this.Sleep(150);
    }

    local town_a = pair[0];
    local town_b = pair[1];
    local tile_a = AITown.GetLocation(town_a);
    local tile_b = AITown.GetLocation(town_b);

    AISign.BuildSign(tile_a, "GPT objective: connect from " + AITown.GetName(town_a));
    AISign.BuildSign(tile_b, "GPT target: " + AITown.GetName(town_b));
    AILog.Info("GPT speedrun objective: connect " + AITown.GetName(town_a) + " to " + AITown.GetName(town_b) + ".");

    local source_build = this.FindBuildableTileNear(tile_a);
    local target_build = this.FindBuildableTileNear(tile_b);

    if (source_build != null) {
        AISign.BuildSign(source_build, "GPT step 1: source road burst");
        this.BuildRoadBurst(source_build, "source");
    } else {
        AISign.BuildSign(tile_a, "GPT could not find source build tiles");
    }

    if (target_build != null) {
        AISign.BuildSign(target_build, "GPT step 2: target road burst");
        this.BuildRoadBurst(target_build, "target");
    } else {
        AISign.BuildSign(tile_b, "GPT could not find target build tiles");
    }

    AISign.BuildSign(tile_b, "GPT speedrun complete: visible construction done");
    AILog.Info("GPT speedrun complete.");

    while (true) {
        this.Sleep(150);
    }
}

function OpenTTDLEBridge::ChooseTownPair()
{
    local towns = AITownList();
    if (towns.Count() < 2) return null;
    towns.Valuate(AITown.GetPopulation);
    towns.Sort(AIAbstractList.SORT_BY_VALUE, false);
    local town_a = towns.Begin();
    towns.RemoveItem(town_a);
    local town_b = towns.Begin();
    return [town_a, town_b];
}

function OpenTTDLEBridge::FindBuildableTileNear(center_tile)
{
    local base_x = AIMap.GetTileX(center_tile);
    local base_y = AIMap.GetTileY(center_tile);
    for (local radius = 2; radius < 24; radius++) {
        for (local dx = -radius; dx <= radius; dx++) {
            for (local dy = -radius; dy <= radius; dy++) {
                local tile = AIMap.GetTileIndex(base_x + dx, base_y + dy);
                if (!AIMap.IsValidTile(tile)) continue;
                if (AITile.IsBuildable(tile)) return tile;
            }
        }
    }
    return null;
}

function OpenTTDLEBridge::BuildRoadBurst(start_tile, label)
{
    AIRoad.SetCurrentRoadType(AIRoad.ROADTYPE_ROAD);

    local x = AIMap.GetTileX(start_tile);
    local y = AIMap.GetTileY(start_tile);
    local built = 0;

    for (local row = 0; row < 5; row++) {
        local previous = null;
        for (local col = 0; col < 10; col++) {
            local tx = x + col;
            local ty = y + row;
            if (row % 2 == 1) tx = x + 9 - col;

            local tile = AIMap.GetTileIndex(tx, ty);
            if (!AIMap.IsValidTile(tile)) {
                previous = null;
                continue;
            }

            if (previous != null && AIMap.DistanceManhattan(previous, tile) == 1) {
                if (this.TryRoad(previous, tile)) {
                    built++;
                    if (built == 1) AISign.BuildSign(tile, "GPT " + label + " build 1");
                    if (built == 8) AISign.BuildSign(tile, "GPT " + label + " build 8");
                    if (built == 18) AISign.BuildSign(tile, "GPT " + label + " build 18");
                    this.Sleep(8);
                }
            }
            previous = tile;
        }
    }

    AISign.BuildSign(start_tile, "GPT " + label + " burst roads built: " + built);
    return built;
}

function OpenTTDLEBridge::TryRoad(tile_a, tile_b)
{
    if (!AIMap.IsValidTile(tile_a) || !AIMap.IsValidTile(tile_b)) return false;
    if (!AITile.IsBuildable(tile_a) && !AIRoad.IsRoadTile(tile_a)) return false;
    if (!AITile.IsBuildable(tile_b) && !AIRoad.IsRoadTile(tile_b)) return false;
    return AIRoad.BuildRoad(tile_a, tile_b);
}
